import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["MerlinParticleHub.tsx"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  minify: false,
  external: [
    "react",
    "react-dom",
    "@react-three/fiber",
    "@react-three/drei",
    "@react-three/postprocessing",
    "three"
  ]
});
