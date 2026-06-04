import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
// `base` is set to the repo name so assets resolve correctly when served from
// GitHub Pages at https://<user>.github.io/occ-js/. Override with the BASE_PATH
// env var (the deploy workflow sets it) or use "/" for local dev / a custom domain.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/occ-js/",
  plugins: [tailwindcss(), react()],
});
