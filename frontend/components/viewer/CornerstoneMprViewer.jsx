'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';

import { assetByAlias, loaderUrlForAsset, mriAssetForSequence } from '@/lib/viewerAssets';

const VIEWPORTS = [
  { id: 'GBM_AXIAL', label: 'AXIAL', orientation: 'AXIAL' },
  { id: 'GBM_CORONAL', label: 'CORONAL', orientation: 'CORONAL' },
  { id: 'GBM_SAGITTAL', label: 'SAGITTAL', orientation: 'SAGITTAL' },
];
const THREE_D_VIEWPORT_ID = 'GBM_3D_REVIEW';

const TOOL_LABELS = { window: 'Window / Level', pan: 'Pan', zoom: 'Zoom' };
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
      try { core.imageLoader.registerImageLoader('nifti', nifti.cornerstoneNiftiImageLoader); } catch {}
      for (const ToolClass of [
        tools.WindowLevelTool,
        tools.PanTool,
        tools.ZoomTool,
        tools.StackScrollTool,
        tools.BrushTool,
        tools.TrackballRotateTool,
      ]) {
        try { tools.addTool(ToolClass); } catch {}
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
  editMode = 'off',
  editSegmentIndex = 2,
  brushSize = 12,
  onEditorController,
  onPotentialEdit,
  reloadToken = 0,
  threeDVisible = false,
  threeDMode = 'composite',
  onThreeDModeChange,
}) {
  const reduceMotion = useReducedMotion();
  const axialRef = useRef(null);
  const coronalRef = useRef(null);
  const sagittalRef = useRef(null);
  const threeDRef = useRef(null);
  const runtimeRef = useRef(null);
  const [state, setState] = useState({ status: 'loading', message: 'Preparing 3D volume…' });
  const [threeDOverlayStatus, setThreeDOverlayStatus] = useState('off');

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
      setState({ status: 'loading', message: `Loading ${sequence} volume and editable AI overlay…` });

      try {
        const { core, tools, nifti } = await bootstrapCornerstone();
        if (cancelled) return;

        const sourceImageIds = await nifti.createNiftiImageIdsAndCacheMetadata({ url: loaderUrlForAsset(sourceAsset) });
        const labelmapImageIds = await nifti.createNiftiImageIdsAndCacheMetadata({ url: loaderUrlForAsset(labelmapAsset) });
        if (cancelled) return;

        const suffix = `${safeIdPart(manifest.study_uuid)}_${sequence}_${sourceAsset.checksum_sha256.slice(0, 8)}_${labelmapAsset.checksum_sha256.slice(0, 8)}`;
        const renderingEngineId = `GBM_ENGINE_${suffix}`;
        const toolGroupId = `GBM_TOOLS_${suffix}`;
        const threeDToolGroupId = `GBM_3D_TOOLS_${suffix}`;
        const volumeId = `nifti:GBM_SOURCE_${suffix}`;
        const labelmapVolumeId = `nifti:GBM_LABEL_VOLUME_${suffix}`;
        const segmentationId = `GBM_LABELMAP_${suffix}`;

        const renderingEngine = new core.RenderingEngine(renderingEngineId);
        const toolGroup = tools.ToolGroupManager.createToolGroup(toolGroupId);
        if (!toolGroup) throw new Error('Cornerstone tool group could not be created.');
        const threeDToolGroup = threeDVisible
          ? tools.ToolGroupManager.createToolGroup(threeDToolGroupId)
          : null;
        if (threeDVisible && !threeDToolGroup) throw new Error('Cornerstone 3D tool group could not be created.');

        for (const ToolClass of [tools.WindowLevelTool, tools.PanTool, tools.ZoomTool, tools.StackScrollTool, tools.BrushTool]) {
          toolGroup.addTool(ToolClass.toolName);
        }

        const elementMap = {
          GBM_AXIAL: axialRef.current,
          GBM_CORONAL: coronalRef.current,
          GBM_SAGITTAL: sagittalRef.current,
        };
        const viewportInputs = VIEWPORTS.map((viewport) => ({
          viewportId: viewport.id,
          element: elementMap[viewport.id],
          type: core.Enums.ViewportType.ORTHOGRAPHIC,
          defaultOptions: {
            orientation: core.Enums.OrientationAxis[viewport.orientation],
            background: [0.015, 0.027, 0.045],
          },
        }));
        if (threeDVisible) {
          viewportInputs.push({
            viewportId: THREE_D_VIEWPORT_ID,
            element: threeDRef.current,
            type: core.Enums.ViewportType.VOLUME_3D,
            defaultOptions: { background: [0.008, 0.018, 0.027] },
          });
        }
        renderingEngine.setViewports(viewportInputs);
        for (const viewport of VIEWPORTS) toolGroup.addViewport(viewport.id, renderingEngineId);
        if (threeDVisible) threeDToolGroup.addViewport(THREE_D_VIEWPORT_ID, renderingEngineId);

        const volume = await core.volumeLoader.createAndCacheVolume(volumeId, { imageIds: sourceImageIds });
        const labelmapVolume = await core.volumeLoader.createAndCacheVolume(labelmapVolumeId, { imageIds: labelmapImageIds });
        await Promise.all([volume.load(), labelmapVolume.load()]);
        await core.setVolumesForViewports(renderingEngine, [{ volumeId }], VIEWPORTS.map((viewport) => viewport.id));
        if (threeDVisible) {
          await core.addVolumesToViewports(renderingEngine, [{ volumeId }], [THREE_D_VIEWPORT_ID]);
          const E = tools.Enums.MouseBindings;
          for (const ToolClass of [tools.TrackballRotateTool, tools.PanTool, tools.ZoomTool]) {
            threeDToolGroup.addTool(ToolClass.toolName);
          }
          threeDToolGroup.setToolActive(tools.TrackballRotateTool.toolName, { bindings: [{ mouseButton: E.Primary }] });
          threeDToolGroup.setToolActive(tools.PanTool.toolName, { bindings: [{ mouseButton: E.Auxiliary }] });
          threeDToolGroup.setToolActive(tools.ZoomTool.toolName, { bindings: [{ mouseButton: E.Secondary }] });
        }

        tools.segmentation.addSegmentations([{
          segmentationId,
          representation: {
            type: tools.Enums.SegmentationRepresentations.Labelmap,
            data: { volumeId: labelmapVolumeId },
          },
        }]);
        await tools.segmentation.addLabelmapRepresentationToViewportMap(
          Object.fromEntries(VIEWPORTS.map((viewport) => [
            viewport.id,
            [{ segmentationId, type: tools.Enums.SegmentationRepresentations.Labelmap }],
          ])),
        );
        for (const viewport of VIEWPORTS) {
          try { tools.segmentation.setActiveSegmentation(viewport.id, segmentationId); } catch {}
        }
        setThreeDOverlayStatus(threeDVisible ? 'loading' : 'off');
        if (threeDVisible) {
          try {
            await tools.segmentation.addLabelmapRepresentationToViewport(
              THREE_D_VIEWPORT_ID,
              [{ segmentationId, type: tools.Enums.SegmentationRepresentations.Labelmap }],
            );
            setThreeDOverlayStatus('ready');
          } catch (threeDError) {
            console.warn('3D segmentation overlay unavailable; MRI volume rendering remains enabled.', threeDError);
            setThreeDOverlayStatus('unavailable');
          }
        }

        try {
          const lut = [
            [0, 0, 0, 0],
            [255, 189, 106, 255],
            [39, 211, 209, 255],
            [140, 140, 140, 0],
            [255, 92, 171, 255],
          ];
          const lutIndex = tools.segmentation.addColorLUT(lut);
          for (const viewport of VIEWPORTS) tools.segmentation.setColorLUT(viewport.id, segmentationId, lutIndex);
          if (threeDVisible) {
            try { tools.segmentation.setColorLUT(THREE_D_VIEWPORT_ID, segmentationId, lutIndex); } catch {}
          }
        } catch {}

        const applyOverlayStyle = (visible, opacity) => {
          for (const viewport of VIEWPORTS) {
            tools.segmentation.setStyle(
              { viewportId: viewport.id, segmentationId, type: tools.Enums.SegmentationRepresentations.Labelmap },
              {
                renderFill: visible,
                renderOutline: visible,
                fillAlpha: visible ? opacity : 0,
                outlineAlpha: visible ? Math.min(1, opacity + 0.35) : 0,
                outlineWidth: 1.5,
              },
            );
          }
          if (threeDVisible && threeDOverlayStatus !== 'unavailable') {
            try {
              tools.segmentation.setStyle(
                { viewportId: THREE_D_VIEWPORT_ID, segmentationId, type: tools.Enums.SegmentationRepresentations.Labelmap },
                {
                  renderFill: visible,
                  renderOutline: visible,
                  fillAlpha: visible ? Math.min(0.34, opacity) : 0,
                  outlineAlpha: visible ? Math.min(0.72, opacity + 0.2) : 0,
                  outlineWidth: 1,
                },
              );
            } catch {}
          }
          renderingEngine.render();
        };

        const applyThreeDMode = (mode) => {
          if (!threeDVisible) return;
          const viewport = renderingEngine.getViewport(THREE_D_VIEWPORT_ID);
          const blendMode = mode === 'mip'
            ? core.Enums.BlendModes.MAXIMUM_INTENSITY_BLEND
            : core.Enums.BlendModes.COMPOSITE;
          viewport.setBlendMode(blendMode);
          viewport.render();
        };

        const resetThreeD = () => {
          if (!threeDVisible) return;
          const viewport = renderingEngine.getViewport(THREE_D_VIEWPORT_ID);
          viewport.resetCamera();
          viewport.render();
        };

        const activateNavigationTool = (toolKey) => {
          const E = tools.Enums.MouseBindings;
          const classes = { window: tools.WindowLevelTool, pan: tools.PanTool, zoom: tools.ZoomTool };
          for (const ToolClass of [...Object.values(classes), tools.StackScrollTool, tools.BrushTool]) {
            try { toolGroup.setToolPassive(ToolClass.toolName); } catch {}
          }
          toolGroup.setToolActive(tools.StackScrollTool.toolName, { bindings: [{ mouseButton: E.Wheel }] });
          toolGroup.setToolActive(tools.PanTool.toolName, { bindings: [{ mouseButton: E.Auxiliary }] });
          toolGroup.setToolActive(tools.ZoomTool.toolName, { bindings: [{ mouseButton: E.Secondary }] });
          if (editMode === 'off') {
            const selected = classes[toolKey] || classes.window;
            toolGroup.setToolActive(selected.toolName, { bindings: [{ mouseButton: E.Primary }] });
          }
        };

        const applyEditTool = (mode, segmentIndex, radius) => {
          activateNavigationTool(activeTool);
          if (mode === 'off') return;
          const strategy = mode === 'erase' ? 'ERASE_INSIDE_CIRCLE' : 'FILL_INSIDE_CIRCLE';
          tools.segmentation.segmentIndex.setActiveSegmentIndex(segmentationId, Number(segmentIndex));
          const existing = toolGroup.getToolConfiguration(tools.BrushTool.toolName) || {};
          toolGroup.setToolConfiguration(
            tools.BrushTool.toolName,
            { ...existing, brushSize: Number(radius), activeStrategy: strategy },
            true,
          );
          toolGroup.setToolActive(tools.BrushTool.toolName, {
            bindings: [{ mouseButton: tools.Enums.MouseBindings.Primary }],
            strategy,
          });
          renderingEngine.render();
        };

        const exportRawLabelmap = () => {
          const vm = labelmapVolume.voxelManager;
          const length = vm.getScalarDataLength();
          const output = new Uint8Array(length);
          for (let index = 0; index < length; index += 1) output[index] = Number(vm.getAtIndex(index) || 0);
          return output;
        };

        applyOverlayStyle(overlayVisible, overlayOpacity);
        applyEditTool(editMode, editSegmentIndex, brushSize);
        applyThreeDMode(threeDMode);
        renderingEngine.render();

        runtimeRef.current = {
          renderingEngine,
          toolGroup,
          segmentationId,
          tools,
          applyOverlayStyle,
          activateNavigationTool,
          applyEditTool,
          applyThreeDMode,
          resetThreeD,
          exportRawLabelmap,
        };
        onEditorController?.({ exportRawLabelmap, segmentationId, labelmapVolumeId });
        setState({ status: 'ready', message: `${sequence} ready` });

        cleanup = () => {
          onEditorController?.(null);
          runtimeRef.current = null;
          try { tools.ToolGroupManager.destroyToolGroup(toolGroupId); } catch {}
          if (threeDVisible) { try { tools.ToolGroupManager.destroyToolGroup(threeDToolGroupId); } catch {} }
          try { tools.segmentation.removeSegmentation(segmentationId); } catch {}
          try { renderingEngine.destroy(); } catch {}
          try { core.cache.removeVolumeLoadObject(volumeId); } catch {}
          try { core.cache.removeVolumeLoadObject(labelmapVolumeId); } catch {}
        };
      } catch (error) {
        if (!cancelled) {
          console.error('Cornerstone viewer initialization failed', error);
          setState({ status: 'error', message: error?.message || 'The medical imaging renderer could not initialize.' });
        }
      }
    }

    setup();
    return () => { cancelled = true; cleanup(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest.study_uuid, sequence, sourceAsset?.checksum_sha256, labelmapAsset?.checksum_sha256, reloadToken, threeDVisible]);

  useEffect(() => { runtimeRef.current?.applyOverlayStyle?.(overlayVisible, overlayOpacity); }, [overlayVisible, overlayOpacity]);
  useEffect(() => { runtimeRef.current?.activateNavigationTool?.(activeTool); runtimeRef.current?.applyEditTool?.(editMode, editSegmentIndex, brushSize); }, [activeTool, editMode, editSegmentIndex, brushSize]);
  useEffect(() => { runtimeRef.current?.applyThreeDMode?.(threeDMode); }, [threeDMode]);
  useEffect(() => {
    const engine = runtimeRef.current?.renderingEngine;
    if (!engine) return;
    for (const viewport of VIEWPORTS) { try { engine.getViewport(viewport.id).resetCamera(); } catch {} }
    if (threeDVisible) { try { engine.getViewport(THREE_D_VIEWPORT_ID).resetCamera(); } catch {} }
    engine.render();
  }, [resetToken]);

  return (
    <section className={editMode === 'off' ? 'medical-viewer-frame' : 'medical-viewer-frame medical-viewer-frame--editing'}>
      <div className="medical-toolbar">
        <div className="tool-cluster" aria-label="Primary interaction tool">
          {Object.entries(TOOL_LABELS).map(([key, label]) => (
            <button
              key={key}
              disabled={editMode !== 'off'}
              className={activeTool === key && editMode === 'off' ? 'tool-button tool-button--active' : 'tool-button'}
              onClick={() => onActiveToolChange(key)}
              title={editMode === 'off' ? `${label} with primary mouse button` : 'Exit correction mode to use this primary tool'}
            >
              <Icon type={key} /><span>{label}</span>
            </button>
          ))}
          <button className="tool-button" onClick={onReset} title="Reset all viewport cameras"><Icon type="reset"/><span>Reset</span></button>
        </div>
        {editMode === 'off' ? (
          <div className="mouse-hints"><span><b>Wheel</b> slices</span><span><b>Right</b> zoom</span><span><b>Middle</b> pan</span></div>
        ) : (
          <div className="edit-live-indicator"><i /> LIVE CORRECTION · LEFT MOUSE {editMode === 'erase' ? 'ERASE' : 'PAINT'}</div>
        )}
      </div>

      <div className={threeDVisible ? 'mpr-grid mpr-grid--fourup' : 'mpr-grid'}>
        {VIEWPORTS.map((viewport) => (
          <motion.div key={viewport.id} className="mpr-panel" initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.99 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3 }}>
            <div className="mpr-panel__label"><span>{viewport.label}</span><small>{sequence}</small></div>
            <div
              ref={viewport.id === 'GBM_AXIAL' ? axialRef : viewport.id === 'GBM_CORONAL' ? coronalRef : sagittalRef}
              className="cornerstone-viewport"
              onContextMenu={(event) => event.preventDefault()}
              onPointerUp={() => { if (editMode !== 'off') onPotentialEdit?.(); }}
            />
            <div className="orientation-corner orientation-corner--top">A</div>
            <div className="orientation-corner orientation-corner--bottom">P</div>
          </motion.div>
        ))}
        {threeDVisible ? (
          <motion.div className="mpr-panel mpr-panel--3d" initial={reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.99 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.3 }}>
            <div className="mpr-panel__label"><span>3D VOLUME</span><small>{sequence}</small></div>
            <div className="three-d-badge">REVIEW ONLY · NOT EDITABLE</div>
            <div ref={threeDRef} className="cornerstone-viewport cornerstone-viewport--3d" onContextMenu={(event) => event.preventDefault()} />
            <div className="three-d-controls" aria-label="3D rendering controls">
              <button className={threeDMode === 'composite' ? 'active' : ''} onClick={() => onThreeDModeChange?.('composite')}>Composite</button>
              <button className={threeDMode === 'mip' ? 'active' : ''} onClick={() => onThreeDModeChange?.('mip')}>MIP</button>
              <button onClick={() => runtimeRef.current?.resetThreeD?.()}>Reset 3D</button>
            </div>
            <div className="three-d-hints"><span><b>Left</b> rotate</span><span><b>Right</b> zoom</span><span><b>Middle</b> pan</span></div>
            <div className={`three-d-overlay-state three-d-overlay-state--${threeDOverlayStatus}`}>
              {threeDOverlayStatus === 'ready' ? 'WT / TC / ET volumetric overlay' : threeDOverlayStatus === 'unavailable' ? 'MRI volume only · 3D labelmap overlay unavailable' : 'Preparing volumetric overlay…'}
            </div>
          </motion.div>
        ) : null}
        {state.status !== 'ready' ? (
          <div className={`viewport-state-overlay viewport-state-overlay--${state.status}`} role="status">
            {state.status === 'loading' ? <div className="tiny-spinner" /> : <span className="error-glyph">!</span>}
            <strong>{state.status === 'loading' ? 'Preparing clinical workspace' : 'Viewer initialization failed'}</strong>
            <span>{state.message}</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}
