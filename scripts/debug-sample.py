import napari
from pathlib import Path


viewer = napari.Viewer()

dw, widget = viewer.window.add_plugin_dock_widget(
        'annotrack', 'sample_from_csv'
        )

root = Path('/Users/jni/data/FindingAndFollowing/annotrack_example/')

widget(viewer,
       path_to_csv=root / 'annotrack_csvs/mini-sample-jni-ome.csv',
       output_dir=root / 'out-annotrack2',
       output_name='result',
       n_samples=15,
       )

napari.run()