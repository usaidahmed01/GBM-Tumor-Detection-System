from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding='utf-8')


def test_viewer_asset_loader_resolves_proxy_url_to_absolute_url():
    content = _read('frontend/lib/viewerAssets.js')
    assert "new URL(proxied, base).href" in content
    assert "window.location?.origin" in content
    assert "proxyApiUrl(asset?.loader_url || asset?.download_url)" in content


def test_cornerstone_receives_validated_absolute_asset_urls():
    content = _read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    assert "const sourceUrl = loaderUrlForAsset(sourceAsset);" in content
    assert "const labelmapUrl = loaderUrlForAsset(labelmapAsset);" in content
    assert "url: sourceUrl" in content
    assert "url: labelmapUrl" in content
    assert "MRI viewer asset URL could not be prepared" in content


def test_viewer_initialization_error_has_in_place_retry():
    content = _read('frontend/components/viewer/CornerstoneMprViewer.jsx')
    assert "viewerRetryToken" in content
    assert "Retry viewer" in content
    assert "setViewerRetryToken((value) => value + 1)" in content


def test_cornerstone_worker_runtime_warning_is_narrowly_suppressed():
    content = _read('frontend/next.config.mjs')
    assert "webpack(?:-runtime)?" in content
    assert "config.ignoreWarnings" in content
