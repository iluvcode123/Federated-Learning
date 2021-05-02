# import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import SimpleITK as sitk

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPooling2D((2, 2))(x)
        return x, p
    else:
        return x

def build_unet(shape, num_classes):
    inputs = Input(shape)

    # Encoder
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 48, pool=True) # 48
    x4, p4 = conv_block(p3, 64, pool=True) # 64
    x5, p5 = conv_block(p4, 128, pool=True) 

    # Bridge
    b1 = conv_block(p5, 256, pool=False) # p4 128

    # Decoder
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x5])
    x6 = conv_block(c1, 128, pool=False)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c2 = Concatenate()([u2, x4])
    x7 = conv_block(c2, 64, pool=False) # 64

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c3 = Concatenate()([u3, x3])
    x8 = conv_block(c3, 48, pool=False) # 48

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x8)
    c4 = Concatenate()([u4, x2])
    x9 = conv_block(c4, 32, pool=False)

    u5 = UpSampling2D((2, 2), interpolation="bilinear")(x9)
    c5 = Concatenate()([u5, x1])
    x10 = conv_block(c5, 16, pool=False)

    # Output layer
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x10)

    return Model(inputs, output)

def dice_coef(pred, truth, smooth=1):

    truth = K.flatten(truth)
    pred = K.flatten(pred)
    intersection = K.sum(K.abs(truth * pred), axis=-1)
    
    return (2. * intersection + smooth) / (K.sum(K.square(truth),-1) + K.sum(K.square(pred),-1) + smooth)

def test_model(test_list, mask_list, model, epoch):
    pred = np.argmax(test_list, axis=-1)
    truth = mask_list

    print("Calculating accuracy...")

    # CATEGORICAL ACC
    acc_m = tf.keras.metrics.CategoricalAccuracy()
    acc_m.update_state(pred, truth)
    acc = acc_m.result().numpy()

    # DICE 
    pred = np.float32(pred) # pred is int, truth is float
    acc_dice = dice_coef(pred, truth)
    print('Dice acc: {:.3%}'.format(acc_dice))

    print('round: {} | categorical_acc: {:.3%} | dice_acc: {:.3%}'.format(epoch+1, acc, acc_dice))
    
    return acc_dice

def create_clients(num_clients=6, initial='client', num_classes=3):
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    client_dict = {}

    for client_idx in range (num_clients):
        train_list = []
        mask_list = []
        
        if client_idx is 0: # BIDMC
            for i in range(0,7):
                if i is 1: continue
                ori_img_filename = 'BIDMC//Case0' + str(i) +'.nii.gz'
                msk_img_filename = 'BIDMC//Case0' + str(i) +'_segmentation.nii.gz'

                train_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
                mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

                for j in range(train_scan.shape[0]):
                    train_list.append(train_scan[j])
                    mask_list.append(mask_scan[j])
        
        if client_idx is 1: # UCL
            for i in range(26,32):
                ori_img_filename = 'UCL//Case' + str(i) +'.nii.gz'
                msk_img_filename = 'UCL//Case' + str(i) +'_segmentation.nii.gz'

                train_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
                mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

                for j in range(train_scan.shape[0]):
                    train_list.append(train_scan[j])
                    mask_list.append(mask_scan[j])
        
        if client_idx is 2: # I2CVB
            for i in range(0,13):
                if i < 10: 
                    ori_img_filename = 'I2CVB//Case0' + str(i) +'.nii.gz'
                    msk_img_filename = 'I2CVB//Case0' + str(i) +'_segmentation.nii.gz'
                else:
                    ori_img_filename = 'I2CVB//Case' + str(i) +'.nii.gz'
                    msk_img_filename = 'I2CVB//Case' + str(i) +'_segmentation.nii.gz'

                train_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
                mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

                for j in range(train_scan.shape[0]):
                    train_list.append(train_scan[j])
                    mask_list.append(mask_scan[j])
        
        if client_idx is 3: # HK
            for i in range(38,43):
                ori_img_filename = 'HK//Case' + str(i) +'.nii.gz'
                msk_img_filename = 'HK//Case' + str(i) +'_segmentation.nii.gz'

                train_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
                mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

                for j in range(train_scan.shape[0]):
                    train_list.append(train_scan[j])
                    mask_list.append(mask_scan[j])

        
        if client_idx is 4: # ISBI_one
            for i in range(0,23):
                if i < 10: 
                    ori_img_filename = 'ISBI_one//Case0' + str(i) +'.nii.gz'
                    msk_img_filename = 'ISBI_one//Case0' + str(i) +'_segmentation.nii.gz'
                else: 
                    ori_img_filename = 'ISBI_one//Case' + str(i) +'.nii.gz'
                    msk_img_filename = 'ISBI_one//Case' + str(i) +'_segmentation.nii.gz'
                
                train_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
                mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

                for j in range(train_scan.shape[0]):
                    train_list.append(train_scan[j])
                    mask_list.append(mask_scan[j])
        
        if client_idx is 5: # ISBI_1.5_one
            for i in range(0,23):
                if i < 10: 
                    ori_img_filename = 'ISBI_1.5_one//Case0' + str(i) +'.nii.gz'
                    msk_img_filename = 'ISBI_1.5_one//Case0' + str(i) +'_Segmentation.nii.gz'
                else: 
                    ori_img_filename = 'ISBI_1.5_one//Case' + str(i) +'.nii.gz'
                    msk_img_filename = 'ISBI_1.5_one//Case' + str(i) +'_Segmentation.nii.gz'
                
                train_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
                mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

                for j in range(train_scan.shape[0]):
                    train_list.append(train_scan[j])
                    mask_list.append(mask_scan[j])
        
        if not train_list: continue

        train_array = np.stack(train_list, axis=0)
        mask_array = np.stack(mask_list, axis=0)

        # Add an extra dimension
        train_array = train_array[:, :, :, np.newaxis]
        mask_array = mask_array[:, :, :, np.newaxis]
        
        # Change mask array to categorical numbers
        mask_array = to_categorical(mask_array, num_classes=num_classes)

        print("Training Array Dimension", client_names[client_idx])
        print(train_array.shape, mask_array.shape)

        client_dict[client_names[client_idx]] = (train_array, mask_array)

    return client_dict

def weight_scaling_factor(clients_dict, client):
    client_img = clients_dict[client][0].shape[0]
    total_img = 0

    for client_name in clients_dict.keys():
        # Return first dimension of 4D array - num of samples
        total_img += clients_dict[client_name][0].shape[0]

    return client_img / total_img

def scale_model_weights(weight, scalar):
    weight_final = []

    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])

    return weight_final

def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    
    # Get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad

def main():
    # Distribute 6 data sets to 6 clients
    num_of_classes = 2
    clients_dict = create_clients(num_classes=num_of_classes)
    clients_names = clients_dict.keys()

    print("Clients:", clients_names)

    # Prepare Data for Testing
    test_list = []
    test_list_mask = []

    # I2CVB
    for i in range(10,13):
        if i < 10: 
            ori_img_filename = 'I2CVB//Case0' + str(i) +'.nii.gz'
            msk_img_filename = 'I2CVB//Case0' + str(i) +'_segmentation.nii.gz'
        else:
            ori_img_filename = 'I2CVB//Case' + str(i) +'.nii.gz'
            msk_img_filename = 'I2CVB//Case' + str(i) +'_segmentation.nii.gz'

        test_scan = sitk.GetArrayFromImage(sitk.ReadImage(ori_img_filename, sitk.sitkFloat32))
        mask_scan = sitk.GetArrayFromImage(sitk.ReadImage(msk_img_filename, sitk.sitkFloat32))

        for j in range(test_scan.shape[0]):
            test_list.append(test_scan[j])
            test_list_mask.append(mask_scan[j])

    # Convert list of np arrays to a numpy array of np arrays
    test_array = np.stack(test_list, axis=0)
    test_array_mask = np.stack(test_list_mask, axis=0)

    test_array = test_array[:, :, :, np.newaxis]

    # Check Dimension
    print("Testing Array Dimension")
    print(test_array.shape, test_array_mask.shape)

    # Initialize model
    img_size_target = 384
    input_layer = ((img_size_target, img_size_target, 1))

    model_global = build_unet(input_layer, num_classes=num_of_classes)
    # model_global.summary()

    loss = 'categorical_crossentropy'
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    optimizer = Adam(1e-4)
    
    epochs = 10
    epochs_local = 50
    
    # Load h5 weights - if model is trained before
    # print("Loading pretrained weights")
    # model_global.load_weights('unet_100.h5')

    # Global training loop
    for epoch in range(epochs):
                
        # Get the global model's weights - serve as the initial weights for all local models
        weights_global = model_global.get_weights()

        model_global.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
        # Initialize for collection of local model weights after scalling
        scaled_local_weight_list = list()
        scaled_global_weight_list = list()

        # Create new local model for each client
        for client in clients_names:

            # Setup model for each client
            img_size_target = 384
            input_layer_local = ((img_size_target, img_size_target, 1))
            model_local = build_unet(input_layer_local, num_classes=num_of_classes)

            model_local.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            
            # Set local model weight to the weight of the global model
            model_local.set_weights(weights_global)
            
            # Fit local model with each client's data
            train_array_local = clients_dict[client][0]
            mask_array_local = clients_dict[client][1]

            print("Training Array Dimension for", client)
            print(train_array_local.shape, mask_array_local.shape)
            
            model_local.fit(train_array_local, mask_array_local, batch_size=8, epochs=epochs_local, verbose=2)

            # Scale the model weights and add to list
            scaling_factor = weight_scaling_factor(clients_dict, client)

            print("Scaling factor", scaling_factor)
            scaled_weights = scale_model_weights(model_local.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            
            # Clear session to free memory after each communication round
            K.clear_session()
            
        # Get the average over all the local model - Take the sum of the scaled weights
        print("Averaging weights of local models...")
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        
        # Update global model 
        print("Setting weights for global model...")
        model_global.set_weights(average_weights)

        # test global model and print out metrics after each communication round
        print("Predicting using global model...")
        pred_global = model_global.predict(test_array)
        
        global_acc = test_model(pred_global, test_array_mask, model_global, epoch)

    # Test for 0s and 1s
    print()
    print("****** Test Data has 0 1 ******")
    count = 0
    count_one = 0

    for ls in test_array_mask:
        if 0 in ls.astype(int): count += 1
        if 1 in ls.astype(int): count_one += 1

    print(count, count_one)

    # save weights
    # print("Saving weights...")
    # model_global.save_weights("unetfednew.h5")

if __name__ == "__main__":
    print("####### START OF TRAINING #######")
    main()
    print("####### END OF RUNNING PROCESS #######")