// Offline cache for the PlantDoctor test page + cross-origin isolation (enables multithreaded WebAssembly).
// Model files are cached by the page itself (Cache "plantdoctor-models-v1") when loaded.
const SHELL = "plantdoctor-shell-v2";
const SHELL_FILES = ["./", "index.html", "sw.js", "ort/ort.webgpu.min.js"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

function isolate(res) {
  if (!res || res.status === 0 || res.type === "opaque") return res;
  const h = new Headers(res.headers);
  h.set("Cross-Origin-Embedder-Policy", "require-corp");
  h.set("Cross-Origin-Opener-Policy", "same-origin");
  h.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  e.respondWith((async () => {
    const models = await caches.open("plantdoctor-models-v1");
    const shell = await caches.open(SHELL);
    const hit = (await models.match(e.request, { ignoreSearch: true })) || (await shell.match(e.request, { ignoreSearch: true }));
    // network first for the page itself (so updates arrive), cache first for everything else
    if (hit && e.request.mode !== "navigate") return isolate(hit);
    try {
      const res = await fetch(e.request);
      if (res.ok && (url.pathname.includes("/ort/") || url.pathname.endsWith("manifest.json") || e.request.mode === "navigate")) {
        shell.put(e.request, res.clone());
      }
      return isolate(res);
    } catch (err) {
      if (hit) return isolate(hit);
      if (e.request.mode === "navigate") { const p = await shell.match("index.html"); if (p) return isolate(p); }
      throw err;
    }
  })());
});
