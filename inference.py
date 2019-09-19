import tensorflow as tf
from utils import *
import os

def get_encoded_vectors(sess, op_dict, folder_path):

    X = np.zeros(shape=(64, 224, 224, 3))
    y = np.zeros(shape=(64,))

    img_index = 0

    all_docs = [doc_img for doc_img in os.listdir(folder_path) if '.DS_Store' not in doc_img][:64]

    for img_name in all_docs:
        img = cv2.imread(os.path.join(folder_path, img_name))
        img = cv2.resize(img, (224, 224))
        X[img_index] = img 
        img_index += 1

    mu, sigma = sess.run([op_dict['mu'], op_dict['sigma']], feed_dict={op_dict['input']: X})
    pass


if __name__ == "__main__":
    sess, op_dict = get_ops('/Users/kartik/personal/tensorflow-generative-model-collections/checkpoint/VAE_documents_64_8/VAE/VAE.model-113')
    folder_path="/Users/kartik/Documents/ami_invoices_sample/" 
    get_encoded_vectors(sess, op_dict, folder_path)
    pass