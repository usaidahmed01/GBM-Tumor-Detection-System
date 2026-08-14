'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';

import { assetByAlias, loaderUrlForAsset, mriAssetForSequence } from '@/lib/viewerAssets';

const VIEWPORTS = [
  { id: 'GBM_AXIAL', label: 'AXIAL', orientation: 'AXIAL' },
  { id: 'GBM_CORONAL', label: 'CORONAL', orientation: 'CORONAL' },
  { id: 'GBM_SAGITTAL', label: 'SAGITTAL', orientation: 'SAGITTAL' },
];

const TOOL_LABELS = {
  window: 'Window / Level',
  pan: 'Pan',
  zoom: 'Zoom',
};

let cornerstoneBootstrapPromise = null;

function Icon({ type }) {
  const common = { width: 17, height: 17, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };
  if (type === 'window') return <svg {...common}><circle cx="12" cy="12" r="7"/><path d="M12 5v14M5 12h14"/><path d="M7.1 7.1 16.9 16.9"/></svg>;
  if (type === 'pan') return <svg {...common}><path d="M8 11V7a2 2 0 0 1 4 0v4"/><path d="M12 10V6a2 2 0 0 1 4 0v6"/><path d="M16 10a2 2 0 0 1 4 0v4c0 4-2.5 7-7 7h-1c-2.5 0-4-1.2-5.2-3L4 14a2 2 0 0 1 3-2.5L8 13"/></svg>;
  if (type === 'zoom') return <svg {...common}><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5M10.5 7.5v6M7.5 10.5h6"/></svg>;
  if (type === 'reset') return <svg {...common}><path d="M4 7v5h5"/><path d="M5.5 16a8 8 0 1 0 .5-9L4 12"/></svg>;
  return null;
}

async function bootstrapCornerstone() {
  if (!cornerstoneBootstrapPromise) {
    cornerstoneBootstrapPromise = (async () => {
      const core = await import('@cornerstonejs/core');
      const tools = await import('@cornerstonejs/tools');
      const nifti = await import('@cornerstonejs/nifti-volume-loader');

      await core.init();
      await tools.init();
      nifti.init();

      // The NIfTI loader is imageId based in Cornerstone3D 5.x.
      // Registering explicitly keeps the contract obvious and is harmless if init already did so.
      try {
        core.imageLoader.registerImageLoader('nifti', nifti.cornerstoneNiftiImageLoader);
      } catch {
        // Already registered during hot reload or loader initialization.
      }

      for (const ToolClass of [tools.WindowLevelTool, tools.PanTool, tools.ZoomTool, tools.StackScrollTool]) {
        try { tools.addTool(ToolClass); } catch { /* already globally registered */ }
      }

      return { core, tools, nifti };
    })();
  }
  return cornerstoneBootstrapPromise;
}

function safeIdPart(value) {
  return String(value || '').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 20);
}

export default function CornerstoneMprViewer({
  manifest,
  sequence,
  overlayVisible,
  overlayOpacity,
  activeTool,
  onActiveToolChange,
  resetToken,
  onReset,
}) {
  const reduceMotion = useReducedMotion();
  const axialRef = useRef(null);
  const coronalRef = useRef(null);
  const sagittalRef = useRef(null);
  const runtimeRef = useRef(null);
  const [state, setState] = useState({ status: 'loading', message: 'Preparing 3D volume…' });

  const sourceAsset = useMemo(() => mriAssetForSequence(manifest, sequence), [manifest, sequence]);
  const labelmapAsset = useMemo(() => assetByAlias(manifest, 'mask_labelmap'), [manifest]);

  useEffect(() => {
    let cancelled = false;
    let cleanup = () => {};

    async function setup() {
      if (!sourceAsset || !labelmapAsset) {
        setState({ status: 'error', message: 'Required MRI or segmentation labelmap asset is unavailable.' });
        return;
      }
      setState({ status: 'loading', message: `Loading ${sequence} volume and AI overlay…` });

      try {
        const { core, tools, nifti } = await bootstrapCornerstone();
        if (cancelled) return;

        const sourceUrl = loaderUrlForAsset(sourceAsset);
        const labelmapUrl = loaderUrlForAsset(labelmapAsset);
        const sourceImageIds = await nifti.createNiftiImageIdsAndCacheMetadata({ url: sourceUrl });
        const labelmapImageIds = await nifti.createNiftiImageIdsAndCacheMetadata({ url: labelmapUrl });
        if (cancelled) return;

        const suffix = `${safeIdPart(manifest.study_uuid)}_${sequence}_${sourceAsset.checksum_sha256.slice(0, 8)}`;
        const renderingEngineId = `GBM_ENGINE_${suffix}`;
        const toolGroupId = `GBM_TOOLS_${suffix}`;
        const volumeId = `nifti:GBM_SOURCE_${suffix}`;
        const segmentationId = `GBM_LABELMAP_${suffix}`;

        const renderingEngine = new core.RenderingEngine(renderingEngineId);
        const toolGroup = tools.ToolGroupManager.createToolGroup(toolGroupId);
        if (!toolGroup) throw new Error('Cornerstone tool group could not be created.');

        for (const ToolClass of [tools.WindowLevelTool, tools.PanTool, tools.ZoomTool, tools.StackScrollTool]) {
          toolGroup.addTool(ToolClass.toolName);
        }

        const elementMap = {
          GBM_AXIAL: axialRef.current,
          GBM_CORONAL: coronalRef.current,
          GBM_SAGITTAL: sagittalRef.current,
        };

        renderingEngine.setViewports(VIEWPORTS.map((viewport) => ({
          viewportId: viewport.id,
          element: elementMap[viewport.id],
          type: core.Enums.ViewportType.ORTHOGRAPHIC,
          defaultOptions: {
            orientation: core.Enums.OrientationAxis[viewport.orientation],
            background: [0.015, 0.027, 0.045],
          },
        })));

        for (const viewport of VIEWPORTS) toolGroup.addViewport(viewport.id, renderingEngineId);

        const volume = await core.volumeLoader.createAndCacheVolume(volumeId, { imageIds: sourceImageIds });
        await volume.load();
        await core.setVolumesForViewports(renderingEngine, [{ volumeId }], VIEWPORTS.map((viewport) => viewport.id));

        tools.segmentation.addSegmentations([
          {
            segmentationId,
            representation: {
              type: tools.Enums.SegmentationRepresentations.Labelmap,
              data: { imageIds: labelmapImageIds },
            },
          },
        ]);

        await tools.segmentation.addLabelmapRepresentationToViewportMap(
          Object.fromEntries(VIEWPORTS.map((viewport) => [
            viewport.id,
            [{ segmentationId, type: tools.Enums.SegmentationRepresentations.Labelmap }],
          ])),
        );

        // BraTS label convention produced by the backend: TC=1, WT=2, ET=4.
        try {
          const lut = [
            [0, 0, 0, 0],
            [255, 189, 106, 255],
            [39, 211, 209, 255],
            [140, 140, 140, 0],
            [255, 92, 171, 255],
          ];
          const lutIndex = tools.segmentation.addColorLUT(lut);
          for (const viewport of VIEWPORTS) {
            tools.segmentation.setColorLUT(viewport.id, segmentationId, lutIndex);
          }
        } catch {
          // Rendering still works with Cornerstone's default LUT if a package patch changes this API.
        }

        const applyOverlayStyle = (visible, opacity) => {
          for (const viewport of VIEWPORTS) {
            tools.segmentation.setStyle(
              {
                viewportId: viewport.id,
                segmentationId,
                type: tools.Enums.SegmentationRepresentations.Labelmap,
              },
              {
                renderFill: visible,
                renderOutline: visible,
                fillAlpha: visible ? opacity : 0,
                outlineAlpha: visible ? Math.min(1, opacity + 0.35) : 0,
                outlineWidth: 1.5,
              },
            );
          }
          renderingEngine.render();
        };

        const activateTool = (toolKey) => {
          const E = tools.Enums.MouseBindings;
          const classes = {
            window: tools.WindowLevelTool,
            pan: tools.PanTool,
            zoom: tools.ZoomTool,
          };
          for (const ToolClass of Object.values(classes)) toolGroup.setToolPassive(ToolClass.toolName);
          toolGroup.setToolPassive(tools.StackScrollTool.toolName);

          toolGroup.setToolActive(tools.StackScrollTool.toolName, { bindings: [{ mouseButton: E.Wheel }] });
          const selected = classes[toolKey] || classes.window;
          const bindings = [{ mouseButton: E.Primary }];
          if (selected !== tools.PanTool) {
            toolGroup.setToolActive(tools.PanTool.toolName, { bindings: [{ mouseButton: E.Auxiliary }] });
          }
          if (selected !== tools.ZoomTool) {
            toolGroup.setToolActive(tools.ZoomTool.toolName, { bindings: [{ mouseButton: E.Secondary }] });
          }
          toolGroup.setToolActive(selected.toolName, { bindings });
        };

        applyOverlayStyle(overlayVisible, overlayOpacity);
        activateTool(activeTool);
        renderingEngine.render();

        runtimeRef.current = {
          renderingEngine,
          toolGroup,
          segmentationId,
          tools,
          applyOverlayStyle,
          activateTool,
        };
        setState({ status: 'ready', message: `${sequence} ready` });

        cleanup = () => {
          runtimeRef.current = null;
          try { tools.ToolGroupManager.destroyToolGroup(toolGroupId); } catch {}
          try { tools.segmentation.removeSegmentation(segmentationId); } catch {}
          try { renderingEngine.destroy(); } catch {}
          try { core.cache.removeVolumeLoadObject(volumeId); } catch {}
        };
      } catch (error) {
        if (!cancelled) {
          console.error('Cornerstone viewer initialization failed', error);
          setState({
            status: 'error',
            message: error?.message || 'The medical imaging renderer could not initialize.',
          });
        }
      }
    }

    setup();
    return () => {
      cancelled = true;
      cleanup();
    };
    // Rebuild only when the source study/sequence changes. Overlay/tool state is updated below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest.study_uuid, sequence, sourceAsset?.checksum_sha256, labelmapAsset?.checksum_sha256]);

  useEffect(() => {
    runtimeRef.current?.applyOverlayStyle?.(overlayVisible, overlayOpacity);
  }, [overlayVisible, overlayOpacity]);

  useEffect(() => {
    runtimeRef.current?.activateTool?.(activeTool);
  }, [activeTool]);

  useEffect(() => {
    const engine = runtimeRef.current?.renderingEngine;
    if (!engine) return;
    for (const viewport of VIEWPORTS) {
      try { engine.getViewport(viewport.id).resetCamera(); } catch {}
    }
    engine.render();
  }, [resetToken]);

  return (
    <section className="medical-viewer-frame">
      <div className="medical-toolbar">
        <div className="tool-cluster" aria-label="Primary interaction tool">
          {Object.entries(TOOL_LABELS).map(([key, label]) => (
            <button
              key={key}
              className={activeTool === key ? 'tool-button tool-button--active' : 'tool-button'}
              onClick={() => onActiveToolChange(key)}
              title={`${label} with primary mouse button`}
            >
              <Icon type={key} />
              <span>{label}</span>
            </button>
          ))}
          <button className="tool-button" onClick={onReset} title="Reset all viewport cameras">
            <Icon type="reset" /><span>Reset</span>
          </button>
        </div>
        <div className="mouse-hints">
          <span><b>Wheel</b> slices</span><span><b>Right</b> zoom</span><span><b>Middle</b> pan</span>
        </div>
      </div>

      <div className="mpr-grid">
        {VIEWPORTS.map((viewport) => (
          <motion.div
            key={viewport.id}
            className="mpr-panel"
            initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.99 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="mpr-panel__label"><span>{viewport.label}</span><small>{sequence}</small></div>
            <div
              ref={viewport.id === 'GBM_AXIAL' ? axialRef : viewport.id === 'GBM_CORONAL' ? coronalRef : sagittalRef}
              className="cornerstone-viewport"
              onContextMenu={(event) => event.preventDefault()}
            />
            <div className="orientation-corner orientation-corner--top">A</div>
            <div className="orientation-corner orientation-corner--bottom">P</div>
          </motion.div>
        ))}

        {state.status !== 'ready' ? (
          <div className={`viewport-state-overlay viewport-state-overlay--${state.status}`} role="status">
            {state.status === 'loading' ? <div className="tiny-spinner" /> : <span className="error-glyph">!</span>}
            <strong>{state.status === 'loading' ? 'Preparing diagnostic workspace' : 'Viewer initialization failed'}</strong>
            <span>{state.message}</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}
