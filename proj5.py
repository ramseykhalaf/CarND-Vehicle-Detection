def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    if color_space != 'RGB':
        img = cv2.cvtColor(img, cv2['COLOR_RGB2'+color_space])
    features = cv2.resize(image, size).ravel()
    return features

def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    example = plt.imread(car_list[0])
    data_dict["image_shape"] = example.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example.dtype
    # Return data_dict
    return data_dict


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            visualise=True, feature_vector=feature_vec,
            block_norm="L2-Hys")
        return features, hog_image
    else:
        features= hog(img, orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            visualise=False, feature_vector=feature_vec,
            block_norm="L2-Hys")
        return features

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for filename in imgs:
        # Read in each one by one
        img = plt.imread(filename)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            img = cv2.cvtColor(img, cv2['COLOR_RGB2'+cspace])

        # Apply bin_spatial() to get spatial color features
        bin_spatial_features = bin_spatial(img, spatial_size)

        # Apply color_hist() to get color histogram features
        color_hist_features = color_hist(img, hist_bins, hist_range)

        # Append the new feature vector to the features list
        features.append( np.concatenate((bin_spatial_features, color_hist_features)))
    # Return list of feature vectors
    return features
