from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import numpy
def num_classfy(N):
	X, y = datasets.load_digits(return_X_y=True)
	clf = LogisticRegression(random_state=0, solver="liblinear", multi_class="auto")
	clf.fit(X,y)
	im = Image.open(N)
	enhancer = ImageEnhance.Brightness(im)
	im_enhanced = enhancer.enhance(2)
	im_gray = im_enhanced.convert(mode="L")
	im_64 = im_gray.resize((8,8))
	im_inverted = ImageOps.invert(im_64)
	X_im2d = numpy.asarray(im_inverted)
	X_im1d = X_im2d.reshape(-1)
	X_multiplied = X_im1d*(16/255)
	print(clf.predict([X_multiplied])[0])