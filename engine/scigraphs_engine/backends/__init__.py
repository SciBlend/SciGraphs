# A backend turns a MeshSpec into draw calls on one graphics API. Nothing is
# imported here: a backend pulls in wgpu and a native library, and
# `import scigraphs_engine` has to keep costing numpy and nothing else. Import
# one by name: `backends.wgpu.Renderer`.
