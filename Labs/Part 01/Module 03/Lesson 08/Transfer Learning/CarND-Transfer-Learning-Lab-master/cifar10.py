import shutil
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tqdm import tqdm


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)



X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=42
)
    

n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)

print(f"Number of train = {n_train}")
print(f"Number of test = {n_test}")
print(f"Number of valid = {n_valid}")

print(X_train.shape)
import cv2



def to_num(new_X):
    new_X = np.array(new_X, np.float32)
    if len(new_X.shape) <= 3:
        new_X = np.expand_dims(new_X, axis=-1)
    return new_X

def standardize(X):
    #X = X/255.0
    X = (X-total_mean)/total_std
    return X

def grayscale(X):
    new_X = []
    for img in X:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        new_X.append(gray)
    return new_X


def yuv(images, layer=0):
    hued = []
    for image in images:
        transit = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        hued.append(transit)
    return hued



#X_train = grayscale(X_train)
#X_valid = grayscale(X_valid)
#X_test = grayscale(X_test)

layer = 0

X_train = yuv(X_train, layer)
X_valid = yuv(X_valid, layer)
X_test = yuv(X_test, layer)

total_mean = np.mean(X_train)
total_std = np.std(X_train)

X_train = standardize(to_num(X_train))
X_valid =  standardize(to_num(X_valid))
X_test =  standardize(to_num(X_test))


print(X_train.shape)

learning_rate = 0.001
epochs = 5
batch_size = 64
input_depth = X_train.shape[-1]
steps_per_epoch = int(np.ceil(n_train / batch_size))
s = 1

from tensorflow.keras import layers
X_train = np.astype(X_train, np.float32)


aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),          # ~10 degrees
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])

def augment_image(image, label):
    image = aug(image)
    return image, label

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#train_data = train_data.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.shuffle(n_train).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

class CNN_Model(tf.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        mu=0
        sigma=0.001
        depthc1 = 16#12
        depthc2 = 32#24
        depthc3 = 64#32
        depthc4 = 128#64
        depthf1 = 1024
        depthf2 = 180
        depthf3 = 43
        self.momentum = 0.9
        self.epsilon = 1e-5
        
        self.wc1 = tf.Variable(tf.random.normal([3, 3, input_depth, depthc1], mean=mu, stddev=sigma), name="weightc1", trainable=True)
        self.bc1 = 0#tf.Variable(tf.zeros([depthc1]), name="biasc1")#, trainable=True)
        self.g1 = tf.Variable(tf.ones([depthc1]), name="g1", trainable=True)
        self.b1  = tf.Variable(tf.zeros(depthc1), name="b1", trainable=True)
        self.mm1 = tf.Variable(tf.ones([depthc1]), name="mm1", trainable=False)
        self.mv1 = tf.Variable(tf.zeros(depthc1), name="mv1", trainable=False)

        self.wc2 = tf.Variable(tf.random.normal([3, 3, depthc1, depthc2], mean=mu, stddev=sigma), name="weightc2", trainable=True)
        self.bc2 = 0#tf.Variable(tf.zeros([depthc2]), name="biasc2")#, trainable=True)
        self.g2 = tf.Variable(tf.ones([depthc2]), name="g2", trainable=True)
        self.b2  = tf.Variable(tf.zeros([depthc2]), name="b2", trainable=True)
        self.mm2 = tf.Variable(tf.ones([depthc2]), name="mm2", trainable=False)
        self.mv2 = tf.Variable(tf.zeros(depthc2), name="mv2", trainable=False)

        self.wc3 = tf.Variable(tf.random.normal([3, 3, depthc2, depthc3], mean=mu, stddev=sigma), name="weightc3", trainable=True)
        self.bc3 = 0#tf.Variable(tf.zeros([depthc3]), name="biasc3")#, trainable=True)
        self.g3 = tf.Variable(tf.ones([depthc3]), name="g3", trainable=True)
        self.b3  = tf.Variable(tf.zeros([depthc3]), name="b3", trainable=True)
        self.mm3 = tf.Variable(tf.ones([depthc3]), name="mm3", trainable=False)
        self.mv3 = tf.Variable(tf.zeros(depthc3), name="mv3", trainable=False)

        self.wc4 = tf.Variable(tf.random.normal([3, 3, depthc3, depthc4], mean=mu, stddev=sigma), name="weightc4", trainable=True)
        self.bc4 = 0#tf.Variable(tf.zeros([depthc4]), name="biasc4")#, trainable=True)
        self.g4 = tf.Variable(tf.ones([depthc4]), name="g4", trainable=True)
        self.b4  = tf.Variable(tf.zeros([depthc4]), name="b4", trainable=True)
        self.mm4 = tf.Variable(tf.ones([depthc4]), name="mm4", trainable=False)
        self.mv4 = tf.Variable(tf.zeros(depthc4), name="mv4", trainable=False)


        self.wfc1 = tf.Variable(tf.random.normal([7*7*(depthc4), depthf1], mean=mu, stddev=sigma), name="weight1", trainable=True)
        self.bfc1 = 0#tf.Variable(tf.zeros([depthf1]), name="bias1", trainable=True)
        self.g_f1 = tf.Variable(tf.ones([depthf1]), name="g_f1", trainable=True)
        self.b_f1  = tf.Variable(tf.zeros([depthf1]), name="b_f1", trainable=True)

        self.wfc2 = tf.Variable(tf.random.normal([depthf1, depthf2], mean=mu, stddev=sigma), name="weight4", trainable=True)
        self.bfc2 = 0#tf.Variable(tf.zeros([depthf2]), name="bias4", trainable=True)
        self.g_f2 = tf.Variable(tf.ones([depthf2]), name="g_f2", trainable=True)
        self.b_f2  = tf.Variable(tf.zeros([depthf2]), name="b_f2", trainable=True)

        self.wfc3 = tf.Variable(tf.random.normal([depthf2, depthf3], mean=mu, stddev=sigma), name="weight2", trainable=True)
        self.bfc3 = tf.Variable(tf.zeros([depthf3]), name="bias2", trainable=True)


    def layer_norm(self, fc, gamma, beta):
        mean, var = tf.nn.moments(fc, axes=[-1], keepdims=True)
        fc = (fc-mean)/tf.sqrt(var+self.epsilon)
        fc = fc * gamma + beta
        return fc



    def batch_norm_train(self, conv, batch_b, batch_g, mm=0, mv=0.01, axes=[0]):

        batch_mean, batch_var = tf.nn.moments(conv, axes=axes)

        mm.assign(self.momentum * mm + (1 - self.momentum) * batch_mean)
        mv.assign(self.momentum * mv + (1 - self.momentum) * batch_var)
        conv = tf.nn.batch_normalization(conv, batch_mean, batch_var, batch_b, batch_g, self.epsilon)
        return conv


    def batch_norm_inf(self, conv, batch_b, batch_g, mm=0, mv=0.01):
        conv = tf.nn.batch_normalization(conv, mm, mv, batch_b, batch_g, self.epsilon)
        return conv
    

    def conv2d(self, feed_data, feed_weight, feed_bias=0, stride=2, padding="VALID", batch_b=1, batch_g=0, is_training=False, mm=0, mv=0.01):
        conv = tf.nn.conv2d(
            feed_data,
            feed_weight,
            strides=[1, stride, stride, 1],
            padding = padding
        )
        #conv = tf.nn.bias_add(conv, feed_bias)
        if is_training:
            conv = self.batch_norm_train(conv, batch_b, batch_g, mm, mv, axes=[0, 1, 2])
        else:
            conv = self.batch_norm_inf(conv, batch_b, batch_g, mm, mv)
        conv = tf.nn.relu(conv)
        return conv


    def maxpool2d(self, feed_data, k=2, s=2):
        max_pool = tf.nn.max_pool(
            feed_data,
            ksize=[1, k, k, 1],
            strides=[1, s, s, 1],
            padding = "SAME"
        )
        return max_pool

        
    def __call__(self, x, is_training=False, return_activations=False):

        """
        Layer 1: 
        3 by 3 with same padding
        3 by 3 with valid padding
        max pool
        Layer 2: 
        3 by 3 with same padding
        3 by 3 with valid padding
        max pool
        Layer 3: 
        Fully Connected Layer
        Layer 4: 
        Fully connected Layer
        Layer 5: 
        Fully Connected Layer
        """
        activations = {}

        
        #Block 1
        conv1 = self.conv2d(x, self.wc1, self.bc1, stride=1, padding="SAME", batch_b=self.b1, batch_g=self.g1, is_training=is_training, mm=self.mm1, mv=self.mv1)
        activations["conv1_1"] = conv1
        conv2 = self.conv2d(conv1, self.wc2, self.bc2, stride=1, batch_b=self.b2, batch_g=self.g2, is_training=is_training, mm=self.mm2, mv=self.mv2)#, padding="SAME")
        activations["conv1_2"] = conv2
        conv3 = self.maxpool2d(conv2, k=2)
        activations["conv1"] = conv3


        # Block 2
        conv4 = self.conv2d(conv3, self.wc3, self.bc3, stride=1, padding="SAME", batch_b=self.b3, batch_g=self.g3, is_training=is_training, mm=self.mm3, mv=self.mv3)
        activations["conv2_1"] = conv4
        conv5 = self.conv2d(conv4, self.wc4, self.bc4, stride=1, batch_b=self.b4, batch_g=self.g4, is_training=is_training, mm=self.mm4, mv=self.mv4)#, padding="SAME")
        activations["conv2_2"] = conv5
        conv6 = self.maxpool2d(conv5, k=2)
        activations["conv2"] = conv6



        flatten = tf.reshape(conv6, [-1, self.wfc1.get_shape().as_list()[0]])
        
        fc1 = tf.nn.bias_add(tf.matmul(flatten, self.wfc1), self.bfc1)
        fc1 = self.layer_norm(fc1, self.g_f1, self.b_f1)
        fc1 = tf.nn.relu(fc1)
        if is_training:
            fc1 = tf.nn.dropout(fc1, rate=0.6)

        fc2 = tf.nn.bias_add(tf.matmul(fc1, self.wfc2), self.bfc2)
        fc2 = self.layer_norm(fc2, self.g_f2, self.b_f2)
        fc2 = tf.nn.relu(fc2) 
        if is_training:
            fc2 = tf.nn.dropout(fc2, rate=0.6)

        out = tf.nn.bias_add(tf.matmul(fc2, self.wfc3), self.bfc3)

        if return_activations:
            return out, activations


        return out


cif_model = CNN_Model()
optimizer = tf.optimizers.Adam(learning_rate)


def cross_entropy(logits, labels):
    labels = tf.cast(labels, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

def run_optimization(batch_X, batch_Y):
    with tf.GradientTape() as g:
        logit = cif_model(batch_X, is_training=True)
        t_loss = cross_entropy(logit, batch_Y)

    gradients = g.gradient(t_loss, cif_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cif_model.trainable_variables))

def test_and_valid(X_data, y_data, length=n_train):
    X_data = np.float32(X_data)
    test_data = tf.data.Dataset.from_tensor_slices((X_data, y_data))
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)  
    move_per_epoch = int(np.ceil(length/batch_size)) 

    pbar = tqdm(test_data.take(move_per_epoch), total=move_per_epoch)
    all_losses = []
    all_accs = []
    for steps, (batch_X, batch_Y) in enumerate(pbar, 1):
        test_logit = cif_model(batch_X, is_training=False)
        test_probs = tf.nn.softmax(test_logit)
        test_loss = cross_entropy(cif_model(batch_X, is_training=True), batch_Y)
        test_accuracy = accuracy(test_probs, batch_Y)
        all_losses.append(test_loss)
        all_accs.append(test_accuracy)


    test_loss = tf.reduce_mean(all_losses)
    test_accuracy = tf.reduce_mean(all_accs)
    return test_loss, test_accuracy



ckpt_dir = "./cif_ckpt"
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=cif_model)#, optimizer=optimizer)

from_scratch = 1

if from_scratch:
    try:
        shutil.rmtree(ckpt_dir)
        print("Checkpoint Deleted")
    except FileNotFoundError:
        print("Path does not exist")
    save_loss = float("inf")
else:
    s = 5 # Last saved training stop
    checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
    valid_logit = cif_model(X_valid, is_training=False)
    valid_probs = tf.nn.softmax(valid_logit)
    valid_loss = cross_entropy(valid_logit, y_valid)
    valid_accuracy = accuracy(valid_probs, y_valid)
    save_loss = valid_loss
    print(f"Valid loss {valid_loss}, Valid accuracy {valid_accuracy}")

manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=3)




for epoch in range(s, epochs+1): 
    pbar = tqdm(train_data.take(steps_per_epoch), total=steps_per_epoch)
    for steps, (batch_X, batch_Y) in enumerate(pbar, 1):
        run_optimization(batch_X, batch_Y)

    train_logit = cif_model(batch_X, is_training=True)
    train_probs = tf.nn.softmax(train_logit)
    train_loss = cross_entropy(train_logit, batch_Y)
    train_accuracy = accuracy(train_probs, batch_Y)

    ##valid_logit = cif_model(X_valid, is_training=False)
    #valid_probs = tf.nn.softmax(valid_logit)
    #valid_loss = cross_entropy(valid_logit, y_valid)
    #valid_accuracy = accuracy(valid_probs, y_valid)
    valid_loss, valid_accuracy = test_and_valid(X_valid, y_valid, n_valid)

    if valid_loss < save_loss:
        save_loss = valid_loss
        saved = manager.save()
        print(f"Model Saved to {saved}")
        

    #if epoch > 15:
    #new_learning_rate = learning_rate / (epoch * 0.5)
    #optimizer.learning_rate.assign(new_learning_rate)

    print(f"Epoch {epoch}, train_loss {train_loss}, train_accuracy {train_accuracy}, valid_loss {valid_loss}, valid_accuracy {valid_accuracy}")
    

test_loss, test_accuracy = test_and_valid(X_test, y_test, n_test)

print(f"Test loss {test_loss}, Test accuracy {test_accuracy}")
