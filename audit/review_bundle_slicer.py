"""Load a downloaded BrainODE review bundle in the 3D Slicer Python console.

Run with: exec(open(r'/local/path/open_in_slicer.py', encoding='utf-8').read())
The file dialog locates bundle_manifest.json when no script path is available.
"""

import json
import math
from pathlib import Path

import qt
import slicer


def load_brainode_review():
    candidate = Path(globals().get('__file__', '')).resolve().parent
    manifest_path = candidate / 'bundle_manifest.json'
    if not manifest_path.is_file():
        selected = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), 'Select bundle_manifest.json in the extracted ZIP',
            '', 'BrainODE manifest (bundle_manifest.json)')
        if not selected:
            return
        manifest_path = Path(selected)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    model_nodes = {}
    segmentation_nodes = {}
    for record in manifest['segmentations']:
        label = slicer.util.loadLabelVolume(str(root / record['file']))
        if label is None:
            raise RuntimeError('Cannot load ' + record['file'])
        seg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', record['name'])
        seg.CreateDefaultDisplayNodes()
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, seg)
        segment = seg.GetSegmentation().GetNthSegment(0)
        segment.SetName('Left hippocampus (source label 75)')
        segment.SetColor(*record['color'])
        # Exact voxel boundaries are supplied as separate models for 3D display.
        display = seg.GetDisplayNode()
        display.SetVisibility3D(False)
        display.SetVisibility2D(record['default_visible'])
        slicer.mrmlScene.RemoveNode(label)
        segmentation_nodes[record['key']] = seg
    for record in manifest['models']:
        node = slicer.util.loadModel(str(root / record['vtk']))
        if node is None:
            raise RuntimeError('Cannot load ' + record['vtk'])
        node.SetName(record['name'])
        display = node.GetDisplayNode()
        display.SetColor(*record['color'])
        display.SetOpacity(record['opacity'])
        display.SetScalarVisibility(False)
        display.SetBackfaceCulling(False)
        display.SetVisibility(record['default_visible'])
        model_nodes[record['key']] = node

    modes = dict(manifest['display_modes'])
    modes.update(baseline=['baseline_mesh'], forecast=['prediction'],
                 reference=['future_mesh'], initial_forecast=['initial_prediction'],
                 segmentation_only=['baseline_voxel_boundary'])

    def show_brainode(mode='segmentation'):
        if mode not in modes:
            raise ValueError('Available modes: ' + ', '.join(modes))
        for key, node in model_nodes.items():
            node.GetDisplayNode().SetVisibility(key in modes[mode])
        for key, node in segmentation_nodes.items():
            node.GetDisplayNode().SetVisibility2D(
                key == ('future_mask' if mode == 'prediction' else 'baseline_mask'))
        slicer.util.forceRenderAllViews()

    def frame_brainode():
        """Fit the shared model envelope once, looking from anatomical left.

        Foreground model bounds avoid fitting the complete 256-cubed mask grid.
        All bundle models participate, including currently hidden comparisons.
        Visibility changes do not move this camera or change magnification.
        """
        bounds = []
        for node in model_nodes.values():
            box = [0.0] * 6
            node.GetRASBounds(box)
            if not all(math.isfinite(x) for x in box) or any(
                    box[2 * axis] > box[2 * axis + 1] for axis in range(3)):
                raise RuntimeError('Invalid model bounds: ' + node.GetName())
            bounds.append(box)
        lower = [min(box[2 * axis] for box in bounds) for axis in range(3)]
        upper = [max(box[2 * axis + 1] for box in bounds) for axis in range(3)]
        center = [(lo + hi) / 2 for lo, hi in zip(lower, upper)]
        extent = [hi - lo for lo, hi in zip(lower, upper)]
        widget = slicer.app.layoutManager().threeDWidget(0)
        view = widget.threeDView()
        view_node = widget.mrmlViewNode()
        view_node.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
        camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(view_node)
        width, height = view.renderWindow().GetSize()
        aspect = max(width, 1) / max(height, 1)
        # Left view: screen horizontal is RAS Y, vertical is RAS Z.
        scale = 1.15 * max(extent[2] / 2, extent[1] / (2 * aspect), 1.0)
        distance = max(4 * math.sqrt(sum(x * x for x in extent)), 100.0)
        camera.SetParallelProjection(True)
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0] - distance, center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        camera.SetParallelScale(scale)
        camera.ResetClippingRange()
        slicer.util.forceRenderAllViews()

    def zoom_brainode(factor=1.5):
        """Increase display magnification while keeping direction and center."""
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError('Zoom factor must be finite and positive')
        view_node = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode()
        camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(view_node)
        if not camera.GetParallelProjection():
            raise RuntimeError('Select Orthographic mode before zoom_brainode')
        camera.SetParallelScale(camera.GetParallelScale() / factor)
        camera.ResetClippingRange()
        slicer.util.forceRenderAllViews()

    globals()['brainode_models'] = model_nodes
    globals()['brainode_segmentations'] = segmentation_nodes
    globals()['show_brainode'] = show_brainode
    globals()['frame_brainode'] = frame_brainode
    globals()['zoom_brainode'] = zoom_brainode
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    slicer.app.processEvents()
    show_brainode('segmentation')
    frame_brainode()
    slicer.util.selectModule('Data')
    print('BrainODE review loaded. No existing scene nodes were removed.')
    print('Modes: show_brainode("segmentation"), show_brainode("change"), '
          'show_brainode("prediction"), show_brainode("models")')
    print('Blue: baseline fitted mesh; orange: future prediction; '
          'green: observed future mesh; gray: exact segmentation voxel boundary.')
    print('Single shapes: show_brainode("baseline"), show_brainode("forecast"), '
          'show_brainode("reference"), show_brainode("segmentation_only").')
    print('Camera stays fixed on switching. zoom_brainode(1.5) enlarges 1.5x; '
          'frame_brainode() restores the shared left-side view.')


load_brainode_review()
