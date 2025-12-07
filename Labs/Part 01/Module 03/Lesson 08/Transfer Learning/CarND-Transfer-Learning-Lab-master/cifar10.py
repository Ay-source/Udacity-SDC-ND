from sklearn.mode_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train,
    test_size=0.3,
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
epochs = 40
batch_size = 64
input_depth = X_train.shape[-1]
steps_per_epoch = int(np.ceil(n_train / batch_size))
s = 1

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


import shutil


ckpt_dir = "./cif_ckpt"
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=model)#, optimizer=optimizer)

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
    for steps, (batch_X, batch_Y) in enumerate(train_data.take(steps_per_epoch), 1):
        run_optimization(batch_X, batch_Y)

    train_logit = cif_model(batch_X, is_training=True)
    train_probs = tf.nn.softmax(train_logit)
    train_loss = cross_entropy(train_logit, batch_Y)
    train_accuracy = accuracy(train_probs, batch_Y)

    valid_logit = cif_model(X_valid, is_training=False)
    valid_probs = tf.nn.softmax(valid_logit)
    valid_loss = cross_entropy(valid_logit, y_valid)
    valid_accuracy = accuracy(valid_probs, y_valid)

    if valid_loss < save_loss:
        save_loss = valid_loss
        saved = manager.save()
        print(f"Model Saved to {saved}")
        

    #if epoch > 15:
    #new_learning_rate = learning_rate / (epoch * 0.5)
    #optimizer.learning_rate.assign(new_learning_rate)

    print(f"Epoch {epoch}, train_loss {train_loss}, train_accuracy {train_accuracy}, valid_loss {valid_loss}, valid_accuracy {valid_accuracy}")
    

test_logit = cif_model(X_test, is_training=False)
test_probs = tf.nn.softmax(test_logit)
test_loss = cross_entropy(model(X_test, is_training=True), y_test)
test_accuracy = accuracy(test_probs, y_test)

print(f"Test loss {test_loss}, Test accuracy {test_accuracy}")