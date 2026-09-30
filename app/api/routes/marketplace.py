"""Installing strategies bought on the Marketplace."""


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.broker.mt5_install import install_compiled_ea

router = APIRouter()

class MarketplaceInstallRequest(BaseModel):
    """A compiled EA bought on the Marketplace - REQ-MKT-1.3.

    The bytes arrive base64-encoded because they travel as JSON from the
    shell, which is the only process holding the licence key needed to
    fetch them. The engine is the only process that knows where MT5 keeps
    its Experts folder, so the two halves meet here.
    """

    filename: str
    content_base64: str
    # Required: the bytes are about to be allowed to trade, and the shell
    # always has the licence server's checksum to send.
    sha256: str


@router.post("/marketplace/install")
def post_marketplace_install(body: MarketplaceInstallRequest) -> dict:
    import base64
    import binascii

    try:
        content = base64.b64decode(body.content_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail="The EA payload is not valid base64.") from error

    result = install_compiled_ea(body.filename, content, body.sha256)
    if not result.installed_paths:
        raise HTTPException(status_code=409, detail=result.detail)
    return {
        "installed_paths": result.installed_paths,
        "detail": result.detail,
    }
