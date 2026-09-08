# PyInstaller spec — design.md B.9. Bundles the Python runtime + engine
# into a single binary consumed by electron-builder's extraResources
# (see app/electron-builder.json5).
# Build: pyinstaller quantengine-engine.spec

block_cipher = None

a = Analysis(
    ["app/main.py"],
    pathex=["."],
    hiddenimports=["uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto", "uvicorn.lifespan", "uvicorn.lifespan.on"],
    hookspath=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="quantengine-engine",
    console=True,
    onefile=True,
)
