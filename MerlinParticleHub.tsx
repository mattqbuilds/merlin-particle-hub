/**
 * MerlinParticleHub.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * A holographic AI command-centre built on React Three Fiber.
 *
 * Features
 * ───────
 *  • Persistent central "MERLIN" text  – Helvetica Bold, gold + cyan rim
 *  • 16 384-particle firefly swarm     – true voice-driven radial ripple
 *  • Arc-reactor waveform              – 12 concentric FFT-driven rings
 *  • Radial tool menu                  – tilt orbit, energy lines, hover glow,
 *                                        transmit pulse, spring-deploy on wakeword
 *  • Holographic response text         – orbits & fades, driven by lastResponse
 *
 * Usage
 * ─────
 *  <MerlinParticleHub
 *    analyser={audioContext.createAnalyser()}
 *    mode="listen"
 *    lastResponse="Understood, Commander."
 *  />
 *
 * Peer dependencies (see package.json):
 *  three, @react-three/fiber, @react-three/drei, @react-three/postprocessing
 * ─────────────────────────────────────────────────────────────────────────────
 */

import {
  useRef,
  useState,
  useEffect,
  useMemo,
  useCallback,
  Fragment,
  useInsertionEffect,
} from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import {
  Text,
  Html,
  Line,
  Billboard,
  Sphere,
} from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

// ─── Types ───────────────────────────────────────────────────────────────────

export type MerlinMode = 'idle' | 'listen' | 'transmit';

export interface ToolDefinition {
  name: string;
  icon: string;
  action: () => void;
}

export interface MerlinParticleHubProps {
  /** Web Audio AnalyserNode; pass null while audio context is not yet ready. */
  analyser: AnalyserNode | null;
  /** Current operating mode – controls menu deploy, pulse and colour states. */
  mode: MerlinMode;
  /** The last text MERLIN spoke; renders as holographic floating reply. */
  lastResponse: string;
  /** Override the default tool set. */
  tools?: ToolDefinition[];
  /** Pixel radius of the canvas (CSS). Defaults to "100%". */
  className?: string;
}

// ─── Default tools ────────────────────────────────────────────────────────────

const DEFAULT_TOOLS: ToolDefinition[] = [
  {
    name: 'Telegram',
    icon: '📱',
    action: () => window.open('https://t.me/your_merlin_bot', '_blank'),
  },
  {
    name: 'Home Asst',
    icon: '🏠',
    action: () => console.warn('[MERLIN] Home Assistant action not wired.'),
  },
  {
    name: 'Browser',
    icon: '🌐',
    action: () => console.log('[MERLIN] Browser automation'),
  },
  {
    name: 'Calendar',
    icon: '📅',
    action: () => console.log('[MERLIN] Calendar'),
  },
  {
    name: 'RAG',
    icon: '🔍',
    action: () => console.log('[MERLIN] RAG query'),
  },
  {
    name: 'Shutdown',
    icon: '⏻',
    action: () => {
      if (window.confirm('Shutdown MERLIN?'))
        fetch('/api/shutdown', { method: 'POST' }).catch(console.error);
    },
  },
];

// ─── Module-level constants ───────────────────────────────────────────────────

// Fix #9 – ArcReactor ring geometry constants hoisted out of the component
// function so they are not re-created on every render.
const RING_COUNT = 12;
const BASE_INNER = 1.88;
const RING_STEP  = 0.14;
const RING_WIDTH = 0.032;

// Fix #4 – Single reusable Vector3 for RadialMenu's lerp target; avoids
// allocating a new object on every animation frame.
const V3_ONE = new THREE.Vector3(1, 1, 1);

// ─── FireflySwarm GPU shaders ─────────────────────────────────────────────────
// Fix #7 – Particle animation moved from a 16 384-iteration JS loop to a
// GLSL vertex shader that runs entirely on the GPU.  The geometry stores the
// immutable base positions; the shader computes ripple + breath each frame
// using two cheap sin() calls per vertex with no CPU upload.

const FIREFLY_VERT = /* glsl */ `
  uniform float uTime;
  uniform float uVolume;
  uniform float uSize;
  uniform float uScale;   // 0.5 * viewport height in px – for size attenuation
  attribute vec3 color;
  varying vec3 vColor;

  void main() {
    vColor = color;

    // Fix #8 – distance computed in the shader (GPU-parallel), replacing the
    // Math.sqrt() call that was executed 16 384 times per frame on the CPU.
    float dist = length(position);

    float ripple = sin(uTime * 22.0 - dist * 7.0) * uVolume * 0.28;
    float breath = sin(uTime * 0.7  + dist * 1.5) * 0.012;
    vec3  pos    = position * (1.0 + ripple + breath);

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);

    // Perspective-correct size attenuation matching THREE.PointsMaterial
    gl_PointSize = uSize * uScale / -mvPosition.z;
    gl_Position  = projectionMatrix * mvPosition;
  }
`;

const FIREFLY_FRAG = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;

  void main() {
    // Discard corners so each point renders as a circle, not a square.
    vec2 uv = gl_PointCoord - 0.5;
    if (dot(uv, uv) > 0.25) discard;
    gl_FragColor = vec4(vColor, uOpacity);
  }
`;

// ─── StatusBadge keyframe – injected once into <head> ────────────────────────
// Fix #11 – The original component injected a <style> tag on every render.
// The keyframe string is now inserted into <head> exactly once via a
// module-level flag checked inside useInsertionEffect.

const PULSE_KEYFRAMES = `@keyframes merlin-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(1.5); }
}`;
let _pulseStyleInjected = false;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Read FFT data into a pre-allocated Uint8Array.
 * Returns the normalised average volume [0, 1].
 */
function readFFT(analyser: AnalyserNode, buf: Uint8Array<ArrayBuffer>): number {
  analyser.getByteFrequencyData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i];
  return sum / buf.length / 255;
}

// ─── Root component ───────────────────────────────────────────────────────────

export default function MerlinParticleHub({
  analyser,
  mode,
  lastResponse,
  tools = DEFAULT_TOOLS,
  className = '',
}: MerlinParticleHubProps) {
  const [isDeployed, setIsDeployed] = useState(false);
  const retractTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (retractTimerRef.current) clearTimeout(retractTimerRef.current);

    if (mode === 'listen' || mode === 'transmit') {
      setIsDeployed(true);
    } else {
      // Keep menu visible for 8 s after returning to idle
      retractTimerRef.current = setTimeout(() => setIsDeployed(false), 8000);
    }

    return () => {
      if (retractTimerRef.current) clearTimeout(retractTimerRef.current);
    };
  }, [mode]);

  return (
    <div
      className={`relative w-full h-full bg-[#080d18] ${className}`}
      style={{ minHeight: 400 }}
    >
      {/*
        Fix #6 – toneMapping is a WebGLRenderer property, not a WebGL context
        attribute.  Passing it inside `gl={{}}` was silently ignored in some
        R3F versions.  onCreated guarantees it is applied to the renderer.
      */}
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
        }}
      >
        <ambientLight intensity={0.12} />
        <pointLight position={[0, 6, 4]} color="#ffd040" intensity={1.4} />
        <pointLight position={[0, -6, 4]} color="#00c8ff" intensity={0.6} />

        {/* ── Central MERLIN wordmark ── */}
        <Text
          fontSize={1.85}
          position={[0, 0, 0]}
          color="#ffd040"
          outlineWidth={0.065}
          outlineColor="#00c8ff"
          anchorX="center"
          anchorY="middle"
          renderOrder={10}
          font="https://fonts.gstatic.com/s/spacegrotesk/v16/V8mDoQDjQSkFtoMM3T6r8E7mF71Q-gowFU.woff2"
        >
          MERLIN
        </Text>

        {/* ── Arc-reactor waveform ── */}
        <ArcReactor analyser={analyser} />

        {/* ── Firefly swarm with voice ripple ── */}
        <FireflySwarm analyser={analyser} mode={mode} />

        {/* ── Radial tool menu (spring-deployed) ── */}
        <DeployableGroup deployed={isDeployed}>
          <RadialMenu radius={3.0} mode={mode} tools={tools} />
        </DeployableGroup>

        {/* ── Holographic floating response ── */}
        <HolographicResponse text={lastResponse} />

        <EffectComposer>
          <Bloom
            luminanceThreshold={0.35}
            luminanceSmoothing={0.75}
            height={512}
            intensity={1.2}
          />
        </EffectComposer>
      </Canvas>

      {/* ── Status badge ── */}
      <StatusBadge mode={mode} />
    </div>
  );
}

// ─── DeployableGroup ──────────────────────────────────────────────────────────
// Pure Three.js spring – no framer-motion-3d dependency required.

interface DeployableGroupProps {
  deployed: boolean;
  children: React.ReactNode;
}

function DeployableGroup({ deployed, children }: DeployableGroupProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const scaleRef = useRef(0);
  // Fix #5 – spring velocity stored in a proper React ref instead of being
  // monkey-patched onto the THREE.Group object as a custom property.
  const velRef = useRef(0);

  useFrame((_, delta) => {
    const target = deployed ? 1 : 0;
    const stiffness = 160;
    const damping   = 22;
    const displacement = scaleRef.current - target;
    const springForce  = -stiffness * displacement;
    const dampingForce = -damping * velRef.current;
    // deltaVel = (F / m) * dt, m = 1
    velRef.current += (springForce + dampingForce) * delta;
    scaleRef.current = Math.max(0, scaleRef.current + velRef.current * delta);
    groupRef.current.scale.setScalar(scaleRef.current);
  });

  return <group ref={groupRef}>{children}</group>;
}

// ─── ArcReactor ───────────────────────────────────────────────────────────────

interface ArcReactorProps {
  analyser: AnalyserNode | null;
}

function ArcReactor({ analyser }: ArcReactorProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const fftRef   = useRef<Uint8Array<ArrayBuffer>>(new Uint8Array(64));

  // Re-allocate when analyser changes
  useEffect(() => {
    fftRef.current = new Uint8Array(analyser?.frequencyBinCount ?? 64);
  }, [analyser]);

  useFrame(() => {
    if (!analyser || !groupRef.current) return;
    readFFT(analyser, fftRef.current);

    groupRef.current.children.forEach((child, i) => {
      const mesh = child as THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
      const bin = Math.floor((i / RING_COUNT) * fftRef.current.length);
      const intensity = fftRef.current[bin] / 255;

      const pulse = 1 + intensity * 0.32;
      mesh.scale.setScalar(pulse);
      mesh.material.opacity = 0.35 + intensity * 0.65;
    });
  });

  const rings = useMemo(
    () =>
      Array.from({ length: RING_COUNT }, (_, i) => {
        const inner = BASE_INNER + i * RING_STEP;
        const outer = inner + RING_WIDTH;
        const color = i % 3 === 0 ? '#00c8ff' : '#ffd040';
        return { inner, outer, color };
      }),
    []
  );

  return (
    <group ref={groupRef} rotation={[Math.PI / 2, 0, 0]}>
      {rings.map(({ inner, outer, color }, i) => (
        <mesh key={i}>
          <ringGeometry args={[inner, outer, 128]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.35}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

// ─── FireflySwarm ─────────────────────────────────────────────────────────────
// 16 384 particles on a sphere shell.
// Fix #7 + #8: animation runs entirely in a GLSL vertex shader – no per-frame
// JS loop, no Math.sqrt() calls on the CPU.  The geometry holds immutable base
// positions; the shader computes ripple + breath using GPU-parallel sin().

interface FireflySwarmProps {
  analyser: AnalyserNode | null;
  mode: MerlinMode;
}

const PARTICLE_COUNT = 16_384;
const SPHERE_RADIUS = 2.4;
const SPHERE_VARIATION = 0.45;

function FireflySwarm({ analyser, mode }: FireflySwarmProps) {
  const pointsRef = useRef<THREE.Points>(null!);
  const fftRef    = useRef<Uint8Array<ArrayBuffer>>(new Uint8Array(64));

  // Base positions are immutable – passed straight to the GPU, never mutated.
  const basePositions = useMemo(() => {
    const arr = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const r     = SPHERE_RADIUS + (Math.random() - 0.5) * SPHERE_VARIATION;
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      arr[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, []);

  const colors = useMemo(() => {
    const arr = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Mostly cyan, occasional gold
      const isGold = Math.random() < 0.08;
      arr[i * 3]     = isGold ? 1.0 : 0.0;
      arr[i * 3 + 1] = isGold ? 0.82 : 0.78;
      arr[i * 3 + 2] = isGold ? 0.25 : 1.0;
    }
    return arr;
  }, []);

  // ShaderMaterial created once; uniforms updated each frame.
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: {
          uTime:    { value: 0 },
          uVolume:  { value: 0 },
          uSize:    { value: 0.015 },
          uScale:   { value: 300 },
          uOpacity: { value: 0.85 },
        },
        vertexShader:   FIREFLY_VERT,
        fragmentShader: FIREFLY_FRAG,
        blending:    THREE.AdditiveBlending,
        depthWrite:  false,
        transparent: true,
      }),
    []
  );

  // Dispose GPU resources when the component unmounts.
  useEffect(() => () => material.dispose(), [material]);

  useEffect(() => {
    fftRef.current = new Uint8Array(analyser?.frequencyBinCount ?? 64);
  }, [analyser]);

  useFrame((state) => {
    if (!pointsRef.current) return;

    const mat = pointsRef.current.material as THREE.ShaderMaterial;
    const t   = state.clock.getElapsedTime();

    mat.uniforms.uTime.value   = t;
    // state.size.height gives CSS pixels; halved to match THREE.PointsMaterial scale
    mat.uniforms.uScale.value  = state.size.height * 0.5;
    mat.uniforms.uSize.value   = mode === 'transmit' ? 0.022 : 0.015;
    mat.uniforms.uVolume.value = analyser ? readFFT(analyser, fftRef.current) : 0;

    // Slow continuous rotation – still done on the CPU since it's just two
    // property writes, not a per-vertex loop.
    pointsRef.current.rotation.y = t * 0.035;
    pointsRef.current.rotation.x = Math.sin(t * 0.12) * 0.06;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        {/* position attribute holds immutable base positions – never re-uploaded */}
        <bufferAttribute attach="attributes-position" args={[basePositions, 3]} />
        <bufferAttribute attach="attributes-color"    args={[colors, 3]} />
      </bufferGeometry>
      <primitive object={material} attach="material" />
    </points>
  );
}

// ─── RadialMenu ───────────────────────────────────────────────────────────────

interface RadialMenuProps {
  radius: number;
  mode: MerlinMode;
  tools: ToolDefinition[];
}

function RadialMenu({ radius, mode, tools }: RadialMenuProps) {
  const groupRef = useRef<THREE.Group>(null!);
  const [hovered, setHovered] = useState<number | null>(null);

  // Fix #12 – one stable handler closure per tool, recreated only when the
  // tools array changes; previously handleOver(i) produced a new closure on
  // every render for every tool item.
  const hoverHandlers = useMemo(
    () => tools.map((_, i) => () => setHovered(i)),
    [tools]
  );
  const handleOut = useCallback(() => setHovered(null), []);

  useFrame((state) => {
    if (!groupRef.current) return;
    const t = state.clock.getElapsedTime();
    groupRef.current.rotation.y += 0.003;

    if (mode === 'transmit') {
      const pulse = 1 + Math.sin(t * 6.5) * 0.1;
      groupRef.current.scale.setScalar(pulse);
    } else {
      // Fix #4 – reuse module-level V3_ONE instead of allocating a new
      // THREE.Vector3(1,1,1) on every frame.
      groupRef.current.scale.lerp(V3_ONE, 0.08);
    }
  });

  const itemData = useMemo(
    () =>
      tools.map((tool, i) => {
        const angle = (i / tools.length) * Math.PI * 2;
        return {
          tool,
          position: new THREE.Vector3(
            Math.cos(angle) * radius,
            Math.sin(angle * 2) * 0.5, // vertical tilt
            Math.sin(angle) * radius
          ),
        };
      }),
    [tools, radius]
  );

  const lineColor = {
    idle:     '#00c8ff',
    listen:   '#44eeff',
    transmit: '#ff5533',
  }[mode];

  return (
    <group ref={groupRef}>
      {itemData.map(({ tool, position }, i) => {
        const isHovered = hovered === i;
        const emissiveColor = isHovered
          ? '#ffd040'
          : mode === 'transmit'
          ? '#ff4422'
          : '#00c8ff';

        return (
          <Fragment key={tool.name}>
            {/* Energy line from centre */}
            <Line
              points={[new THREE.Vector3(0, 0, 0), position]}
              color={lineColor}
              lineWidth={isHovered ? 2.5 : 1.4}
              transparent
              opacity={isHovered ? 0.9 : 0.4}
            />

            {/* Sphere node */}
            <Billboard
              position={position}
              onPointerOver={hoverHandlers[i]}
              onPointerOut={handleOut}
              onClick={tool.action}
            >
              <Sphere args={[0.2, 32, 32]}>
                <meshStandardMaterial
                  color="#030810"
                  emissive={emissiveColor}
                  emissiveIntensity={isHovered ? 2.2 : mode === 'transmit' ? 1.2 : 0.7}
                  metalness={0.95}
                  roughness={0.15}
                />
              </Sphere>

              {/* Icon + label via HTML overlay */}
              <Html center distanceFactor={10}>
                <div
                  className="pointer-events-none select-none text-center"
                  style={{
                    transform: `scale(${isHovered ? 1.25 : 1})`,
                    transition: 'transform 0.15s ease',
                  }}
                >
                  <div
                    style={{
                      fontSize: 22,
                      filter: `drop-shadow(0 0 8px ${emissiveColor})`,
                    }}
                  >
                    {tool.icon}
                  </div>
                  <div
                    style={{
                      fontSize: 9,
                      fontFamily: 'monospace',
                      letterSpacing: '1px',
                      color: emissiveColor,
                      textShadow: `0 0 8px ${emissiveColor}`,
                      marginTop: 3,
                      opacity: isHovered ? 1 : 0.6,
                      transition: 'opacity 0.15s ease',
                    }}
                  >
                    {tool.name.toUpperCase()}
                  </div>
                </div>
              </Html>
            </Billboard>
          </Fragment>
        );
      })}
    </group>
  );
}

// ─── HolographicResponse ─────────────────────────────────────────────────────
// Fades in, orbits slowly, then fades out after a timeout.
//
// Fixes applied:
//  #1  – `visible` state drives mount/unmount so the component actually
//         disappears from the scene graph and its useFrame stops running.
//         The original ref-only approach meant the guard `return null` never
//         re-evaluated after the first render.
//  #2  – fillOpacity / outlineOpacity are set entirely imperatively from
//         useFrame.  The conflicting JSX props have been removed so React's
//         reconciler can't reset them on re-renders.
//  #3  – Dead `opacity` / `setOpacity` state removed.
//  #10 – useFrame bails out immediately when the component is not visible.

interface HolographicResponseProps {
  text: string;
}

function HolographicResponse({ text }: HolographicResponseProps) {
  const meshRef    = useRef<THREE.Mesh>(null!);
  const [displayed, setDisplayed] = useState('');
  const [visible,   setVisible]   = useState(false);
  const opacityRef = useRef(0);
  const phaseRef   = useRef<'in' | 'hold' | 'out' | 'hidden'>('hidden');
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const MAX_CHARS = 64;

  useEffect(() => {
    if (!text || text.length < 3) return;

    if (holdTimerRef.current) clearTimeout(holdTimerRef.current);

    setDisplayed(
      text.length > MAX_CHARS ? text.substring(0, MAX_CHARS - 1) + '…' : text
    );

    // Reset opacity so a re-triggered response always fades in cleanly.
    opacityRef.current = 0;
    phaseRef.current   = 'in';
    setVisible(true);

    holdTimerRef.current = setTimeout(() => {
      phaseRef.current = 'out';
    }, 4200);

    return () => {
      if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    };
  }, [text]);

  useFrame((state, delta) => {
    // Fix #10 – skip all work while not visible.
    if (!visible) return;

    const phase = phaseRef.current;

    if (phase === 'in') {
      opacityRef.current = Math.min(1, opacityRef.current + delta * 2.5);
      if (opacityRef.current >= 1) phaseRef.current = 'hold';
    } else if (phase === 'out') {
      opacityRef.current = Math.max(0, opacityRef.current - delta * 1.2);
      if (opacityRef.current <= 0) {
        phaseRef.current = 'hidden';
        // Trigger re-render so the component unmounts and useFrame stops.
        setVisible(false);
      }
    }

    if (meshRef.current) {
      // Fix #2 – set troika properties imperatively; no JSX prop conflicts.
      (meshRef.current as any).fillOpacity    = opacityRef.current * 0.82;
      (meshRef.current as any).outlineOpacity = opacityRef.current * 0.4;
      // Slow orbit
      const t = state.clock.getElapsedTime();
      meshRef.current.position.y = 2.6 + Math.sin(t * 0.6) * 0.12;
      meshRef.current.rotation.y = Math.sin(t * 0.3) * 0.08;
    }
  });

  // Fix #1 – state-driven mount/unmount: the component is truly removed from
  // the scene when not visible, so useFrame stops executing.
  if (!visible) return null;

  return (
    <Text
      ref={meshRef}
      fontSize={0.6}
      position={[0, 2.6, 0]}
      color="#00e8ff"
      anchorX="center"
      anchorY="middle"
      outlineWidth={0.018}
      outlineColor="#ffffff"
      maxWidth={4.5}
      textAlign="center"
    >
      {displayed}
    </Text>
  );
}

// ─── StatusBadge ─────────────────────────────────────────────────────────────

interface StatusBadgeProps {
  mode: MerlinMode;
}

function StatusBadge({ mode }: StatusBadgeProps) {
  // Fix #11 – keyframe injected into <head> exactly once (module-level flag
  // prevents duplicate <style> tags across re-renders and StrictMode double
  // invocations).  useInsertionEffect runs before any DOM mutations.
  useInsertionEffect(() => {
    if (_pulseStyleInjected) return;
    const style = document.createElement('style');
    style.textContent = PULSE_KEYFRAMES;
    document.head.appendChild(style);
    _pulseStyleInjected = true;
  }, []);

  const label = {
    idle:     'STANDBY',
    listen:   'LISTENING',
    transmit: 'TRANSMITTING',
  }[mode];

  const dotColor = {
    idle:     '#5588aa',
    listen:   '#00e8ff',
    transmit: '#ff5533',
  }[mode];

  return (
    <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-2 pointer-events-none">
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: dotColor,
          boxShadow: `0 0 8px ${dotColor}`,
          display: 'inline-block',
          animation: mode !== 'idle' ? 'merlin-pulse 1.1s ease-in-out infinite' : undefined,
        }}
      />
      <span
        style={{
          color: dotColor,
          fontFamily: 'monospace',
          fontSize: 11,
          letterSpacing: '3px',
          textShadow: `0 0 12px ${dotColor}`,
        }}
      >
        {label}
      </span>
    </div>
  );
}
