import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const source = dirname(fileURLToPath(import.meta.url));
const output = resolve(source, "dist");
await rm(output, { recursive: true, force: true });
await mkdir(resolve(output, "assets"), { recursive: true });
await Promise.all([
  cp(resolve(source, "index.html"), resolve(output, "index.html")),
  cp(resolve(source, "app.js"), resolve(output, "assets", "app.js")),
  cp(resolve(source, "styles.css"), resolve(output, "assets", "styles.css")),
  cp(resolve(source, "favicon.svg"), resolve(output, "assets", "favicon.svg")),
  cp(resolve(source, "static_headers"), resolve(output, "_headers")),
]);
console.log(`Build estatico criado em ${output}`);
