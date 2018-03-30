import matplotlib
import sys

# Do not display GUI only when generating output
if __name__ != '__main__' or len(sys.argv) >= 3:
  matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.decomposition

# Internal
import dataset
import model as gradtype_model

COLOR_MAP = matplotlib.colors.ListedColormap([
  [ 0.25882352941176473, 0.6470588235294118, 0.3215686274509804 ],
  [ 0.6784313725490196, 0.2980392156862745, 0.8705882352941177 ],
  [ 0.3058823529411765, 0.6588235294117647, 0.1607843137254902 ],
  [ 0.36470588235294116, 0.21176470588235294, 0.7529411764705882 ],
  [ 0.5176470588235295, 0.6235294117647059, 0.1450980392156863 ],
  [ 0.8549019607843137, 0.2823529411764706, 0.788235294117647 ],
  [ 0.29411764705882354, 0.44313725490196076, 0.09019607843137255 ],
  [ 0.396078431372549, 0.396078431372549, 0.8392156862745098 ],
  [ 0.6745098039215687, 0.5803921568627451, 0.15294117647058825 ],
  [ 0.5568627450980392, 0.18823529411764706, 0.5568627450980392 ],
  [ 0.615686274509804, 0.6274509803921569, 0.27450980392156865 ],
  [ 0.8823529411764706, 0.2549019607843137, 0.6039215686274509 ],
  [ 0.2901960784313726, 0.615686274509804, 0.45098039215686275 ],
  [ 0.8745098039215686, 0.20392156862745098, 0.2196078431372549 ],
  [ 0.1607843137254902, 0.6313725490196078, 0.6 ],
  [ 0.9176470588235294, 0.3176470588235294, 0.13333333333333333 ],
  [ 0.3411764705882353, 0.5254901960784314, 0.8156862745098039 ],
  [ 0.8666666666666667, 0.4666666666666667, 0.1568627450980392 ],
  [ 0.2901960784313726, 0.2980392156862745, 0.5725490196078431 ],
  [ 0.807843137254902, 0.5607843137254902, 0.16470588235294117 ],
  [ 0.611764705882353, 0.5098039215686274, 0.8313725490196079 ],
  [ 0.4549019607843137, 0.6, 0.30980392156862746 ],
  [ 0.8784313725490196, 0.24313725490196078, 0.41568627450980394 ],
  [ 0.19607843137254902, 0.4, 0.19215686274509805 ],
  [ 0.7725490196078432, 0.4470588235294118, 0.7490196078431373 ],
  [ 0.4627450980392157, 0.4196078431372549, 0.10588235294117647 ],
  [ 0.4745098039215686, 0.24705882352941178, 0.49019607843137253 ],
  [ 0.6705882352941176, 0.45098039215686275, 0.17647058823529413 ],
  [ 0.28627450980392155, 0.596078431372549, 0.788235294117647 ],
  [ 0.6705882352941176, 0.23529411764705882, 0.11764705882352941 ],
  [ 0.5529411764705883, 0.5215686274509804, 0.28627450980392155 ],
  [ 0.6549019607843137, 0.17647058823529413, 0.38823529411764707 ],
  [ 0.3607843137254902, 0.2980392156862745, 0.09411764705882353 ],
  [ 0.8627450980392157, 0.4549019607843137, 0.615686274509804 ],
  [ 0.7607843137254902, 0.5764705882352941, 0.37254901960784315 ],
  [ 0.5568627450980392, 0.27058823529411763, 0.403921568627451 ],
  [ 0.8745098039215686, 0.45098039215686275, 0.3411764705882353 ],
  [ 0.5372549019607843, 0.21176470588235294, 0.23137254901960785 ],
  [ 0.5725490196078431, 0.3803921568627451, 0.20392156862745098 ],
  [ 0.6941176470588235, 0.1803921568627451, 0.22745098039215686 ],
  [ 0.5411764705882353, 0.26666666666666666, 0.10980392156862745 ],
  [ 0.8196078431372549, 0.45098039215686275, 0.4549019607843137 ]
])

def to_color(index):
  return index

def pca(train_coords, validate_coords, fname):
  fig = plt.figure(1, figsize=(8, 6))
  ax = Axes3D(fig, elev=-150, azim=110)
  pca = sklearn.decomposition.PCA(n_components=3, random_state=0x7ed1ae6e)

  ax.set_xlim(left=-1.2, right=1.2)
  ax.set_ylim(bottom=-1.2, top=1.2)
  ax.set_zlim(bottom=-1.2, top=1.2)

  # Fit coordinates
  pca.fit(np.concatenate(train_coords))
  pca.fit(np.concatenate(validate_coords))

  # Transform coordinates and print labels
  legend = []
  for i in range(0, len(dataset.LABELS)):
    color = COLOR_MAP(to_color(i))
    legend.append(mpatches.Patch(color=color, label=dataset.LABELS[i]))
  ax.legend(handles=legend, fontsize=8)

  all_coords = [ train_coords, validate_coords ]
  for coord_type in range(0, len(all_coords)):
    colors = []
    all_x = []
    all_y = []
    all_z = []

    coords = all_coords[coord_type]
    is_train = coord_type is 0
    for i in range(0, len(coords)):
      ds_coords = pca.transform(coords[i])

      x = ds_coords[:, 0]
      y = ds_coords[:, 1]
      z = ds_coords[:, 2]

      color = COLOR_MAP(to_color(i))

      colors += [ color ] * len(x)
      all_x.append(x)
      all_y.append(y)
      all_z.append(z)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    marker = 'o' if is_train else '^'
    size = 5 if is_train else 8
    ax.scatter(all_x, all_y, all_z, c=colors, marker=marker,
               edgecolor='k', s=size, alpha=0.75, linewidths=0.0,
               edgecolors='none')

  if fname == None:
    plt.show()
  else:
    plt.savefig(fname=fname)
    print("Saved image to " + fname)

if __name__ == '__main__':
  import sys

  datasets, sequence_len = dataset.parse()
  siamese, _, _ = gradtype_model.create(sequence_len)
  siamese.load_weights(sys.argv[1])

  train_datasets, validate_datasets = dataset.split(datasets)
  train_coords = dataset.evaluate_model(siamese, train_datasets)
  validate_coords = dataset.evaluate_model(siamese, validate_datasets)
  fname = sys.argv[2] if len(sys.argv) >= 3 else  None
  pca(train_coords, validate_coords, fname)
