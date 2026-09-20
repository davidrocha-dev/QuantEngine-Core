# PyInstaller spec — design.md B.9. Bundles the Python runtime + engine
# into a single binary consumed by electron-builder's extraResources
# (see app/electron-builder.config.cjs).
# Build: pyinstaller quantengine-engine.spec

block_cipher = None

a = Analysis(
    # run_engine.py, not app/main.py: the ASGI module has no entry point,
    # and the frozen worker processes re-run this script (freeze_support).
    ["run_engine.py"],
    pathex=["."],
    # "app.main" is imported by uvicorn from a string ("app.main:app"), so
    # static analysis cannot see it; the websockets protocol backs WS /events.
    hiddenimports=[
        "app.main",
        "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
        "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan", "uvicorn.lifespan.on",
    ],
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
