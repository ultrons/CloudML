import tensorflow as tf
from PIL import Image
import numpy as np
import json, os
import argparse
from io import BytesIO

parser=argparse.ArgumentParser()
parser.add_argument(
        '--image',
        help="input image",
        required=True
        )

args = parser.parse_args()
#with tf.gfile.FastGFile(args.image, 'r') as ifp:
#  image_data = ifp.read()
import base64
'''
with open(args.image, "rb") as imageFile: 
    im = base64.b64encode(imageFile.read())
    im=Image.open(im)
    im=im.resize((299,299), Image.NEAREST)
    buffered=BytesIO()
    im.save(buffered, format='JPEG')
    im_str=base64.b64encode(buffered.getvalue())
    '''

im=Image.open(args.image)
im=im.resize((299,299), Image.NEAREST)
buffered=BytesIO()
im.save(buffered, format='JPEG')
im_str=base64.b64encode(buffered.getvalue(), '-_')

# Testing the serving transformations
sess = tf.Session()
def _preprocess_image(image_bytes): 
    image = tf.decode_base64(image_bytes)
   # image = tf.image.decode_jpeg(image, 3)
    image = tf.image.decode_image(image, 3)
    return image
image_bytes_list = tf.placeholder(dtype=tf.string)
#images = tf.map_fn(
#                  _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
images=_preprocess_image(image_bytes_list)
images = tf.reshape(images,[299,299,3]) # Channel is assumed to be constant

image_o = sess.run(
        images, feed_dict={image_bytes_list: im_str})

print(image_o.shape)








image_dict={"image_bytes": im_str}

targetOut=os.path.basename(args.image).replace(".jpg", ".json")
with open(targetOut, 'w') as f:
    json.dump(image_dict, f)

        

