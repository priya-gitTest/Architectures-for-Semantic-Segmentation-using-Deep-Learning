from keras import backend as K


def dice_coef(y_true, y_pred):
	"""
	Dice coefficient between the predicted segmentation map 
	and the true segmentation map.

	PARAMETERS
	----------

	y_true: tensor,
			true segmentation map

	y_pred: tensor,
			predicted segmentation map


	RETURNS
	-------

	dice_coef: tensor,
			   Calculated dice coefficient

	"""

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	dice_coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
	return dice_coef


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)
