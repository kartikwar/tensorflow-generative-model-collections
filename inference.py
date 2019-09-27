import tensorflow as tf
from utils import *
import os
import json
import numpy as np
import math

def get_encoded_vectors(sess, op_dict, folder_path):

    encoded_vectors = {}

    all_docs = [doc_img for doc_img in os.listdir(folder_path) if '.DS_Store' not in doc_img]

    iterations = int(math.ceil(len(all_docs) / 64.0))

    for iteration in range(iterations):

        docs = all_docs[iteration * 64 : (iteration + 1) * 64] 

        X = np.zeros(shape=(64, 224, 224, 3))
        y = np.zeros(shape=(64,))

        img_index = 0

    

        for img_name in docs:
            img = cv2.imread(os.path.join(folder_path, img_name))
            img = cv2.resize(img, (224, 224))
            X[img_index] = img 
            img_index += 1

        mu, sigma = sess.run([op_dict['mu'], op_dict['sigma']], feed_dict={op_dict['input']: X})

        for index in range(len(docs)):
            im_name = docs[index]
            mu_ = mu[index]
            encoded_vectors[im_name] = mu_.tolist()

    return encoded_vectors


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a-b)
    return dist

def debug():
    with open('template_vectors.json', 'r') as vecf:
        vectors = json.load(vecf)
    # dist1 = calculate_distance(vectors["7_35"], vectors["7_42"])
    dist2 = calculate_distance(vectors["020513314_1812.pdf0.jpg"], vectors["020513314_1812.pdf2.jpg"])
    dist3 = calculate_distance(vectors["032147968_1901.pdf0.jpg"], vectors["032147799_1905.pdf0.jpg"])
    dist4 = calculate_distance(vectors["pdf_page-0002.jpg"], vectors["pdf_page-0003.jpg"])
    dist5 = calculate_distance(vectors["sample_page-0001.jpg"], vectors["sample_page-0002.jpg"])
    dist6 = calculate_distance(vectors["sample_page-0001.jpg"], vectors["pdf_page-0003.jpg"])
    pass

def run_inference():
    sess, op_dict = get_ops('/Users/kartik/personal/tensorflow-generative-model-collections/checkpoint/VAE_documents_64_8/VAE/VAE.model-113')
    folder_path="/Users/kartik/Documents/ami_invoices_sample/" 
    vectors = get_encoded_vectors(sess, op_dict, folder_path)
    with open('template_vectors.json', 'w') as tv:
        json.dump(vectors, tv)

if __name__ == "__main__":
    # run_inference()
    debug()