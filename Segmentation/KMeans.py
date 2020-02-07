class KmeansSegmentation:
    def segmentation_grey(self, image, k, image_name):
        """Performs segmentation of an grey level input image using KMeans algorithm, using the intensity of the pixels as features
        takes as input:
        image: a grey scale image
        return an segemented image
        -----------------------------------------------------
        Sample implementation for K-means
        1. Initialize cluster centers
        2. Assign pixels to cluster based on (intensity) proximity to cluster centers
        3. While new cluster centers have moved:
            1. compute new cluster centers based on the pixels
            2. Assign pixels to cluster based on the proximity to new cluster centers

        """
        import matplotlib.pyplot as plt
        import cv2
        import random
        import os
        import numpy as np
        #random.seed(152205)
        # image = cv2.imread(image, 0)
        height, width = image.shape
        centroids = []
        clusters = []
        for i in range(k):
            centroids.append(random.randint(0,255))
            clusters.append([])
        print("initialized centroids", centroids)
        updateI = 0
        J = []
        while 1:
            #forms K clusters:assign each pixel to the closest centroid; proximity: pixel_intensity minus centroids' intensity
            Jsum_inside_clusters = np.zeros(k)
            for m in range(height):
                for n in range(width):
                    DistanceToCentroids = []
                    for i in range(k):
                        DistanceToCentroids.append(abs(image[m, n]-centroids[i]))
                    for i in range(k):
                        if DistanceToCentroids.index(min(DistanceToCentroids)) == i:
                            clusters[i].append([m, n])
                            Jsum_inside_clusters[i] = Jsum_inside_clusters[i] + DistanceToCentroids[i]
                        else:
                            pass
            Jsum_all_clusters_old = sum(Jsum_inside_clusters)
            #update centroids
            clusters_intensity_sum = []
            cluster_intensity_average = []
            for i in range(k):
                clusters_intensity_sum.append(0)
                for j in range(len(clusters[i])):
                    clusters_intensity_sum[i] = clusters_intensity_sum[i] + image[clusters[i][j][0], clusters[i][j][1]]
                cluster_intensity_average.append(clusters_intensity_sum[i]/len(clusters[i]))
                centroids[i]= cluster_intensity_average[i]
            updateI = updateI + 1
            print("update centroids! ", updateI)
            for i in range(k):
                print("No.",i ,"cluster's centroid:", centroids[i])

            # calculate objective function J after update centroids
            Jsum_inside_clusters = np.zeros(k)
            for m in range(height):
                for n in range(width):
                    DistanceToCentroids = []
                    for i in range(k):
                        DistanceToCentroids.append(abs(image[m, n]-centroids[i]))
                    for i in range(k):
                        if DistanceToCentroids.index(min(DistanceToCentroids)) == i:
                            Jsum_inside_clusters[i] = Jsum_inside_clusters[i] + DistanceToCentroids[i]
                        else:
                            pass
            Jsum_all_clusters_new = sum(Jsum_inside_clusters)
            J.append(Jsum_all_clusters_new)
            if abs(Jsum_all_clusters_old-Jsum_all_clusters_new) <= 1:
                break
            else:
            #flush clusters for assignment with new centroids
                clusters = []
                for i in range(k):
                    clusters.append([])

        plt.figure()
        plt.title(image_name + " GRAY Objective function J")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.text(max(list(range(updateI))), max(J), "K = "+str(k))
        plt.plot(list(range(updateI)), J)
        path = os.getcwd()# get current path
        output_path = path + '/output/'
        plt.savefig(output_path + "GRAY_Loss for K="+str(k)+" " + image_name +'.jpg')
        for i in range(k):
            for [index,coordinates] in enumerate(clusters[i]):
                image[coordinates[0], coordinates[1]] = centroids[i]
        cv2.imwrite(output_path+str(k)+'_GRAY_After_Segmentation_'+image_name+'.jpg', image)
        return image

# segmentation_rgb(self = True, image = image_path, k = 2)
    def segmentation_rgb(self, image, k, image_name):
        """Performs segmentation of a color input image using KMeans algorithm, using the intensity of the pixels (R, G, B)
        as features
        takes as input:
        image: a color image
        return an segemented image"""
        import matplotlib.pyplot as plt
        import cv2
        import random
        import numpy as np
        import os
        #random.seed(1881448)
        # image = cv2.imread(image)
        height, width, channel = image.shape
        centroids = []
        clusters = []
        for i in range(k):
            centroids.append([])
            for j in range(channel):
                centroids[i].append(random.randint(0, 255))
            clusters.append([])
        print("initialized centroids", centroids)
        updateI = 0
        J = []
        while 1:
            # forms K clusters:assign each pixel to the closest centroid; proximity: pixel_intensity minus centroids' intensity
            Jsum_inside_clusters = np.zeros(k)
            for m in range(height):
                for n in range(width):
                    DistanceToCentroids = []
                    for i in range(k):
                        DistanceToCentroids.append(np.sqrt(int(image[m, n, 0] - centroids[i][0])**2 + int(image[m, n, 1] - centroids[i][1])**2 +int(image[m, n, 2] - centroids[i][2])**2))
                    for i in range(k):
                        if DistanceToCentroids.index(min(DistanceToCentroids)) == i:
                            clusters[i].append([m, n])
                            Jsum_inside_clusters[i] = Jsum_inside_clusters[i] + DistanceToCentroids[i]
                        else:
                            pass
            Jsum_all_clusters_old = sum(Jsum_inside_clusters)
            # update centroids
            clusters_intensity_sum = []
            cluster_intensity_average = []
            for i in range(k):
                clusters_intensity_sum.append([])
                cluster_intensity_average.append([])
                for c in range(channel):
                    clusters_intensity_sum[i].append(0)
                    cluster_intensity_average[i].append(0)
            for i in range(k):
                for j in range(len(clusters[i])):
                    clusters_intensity_sum[i][0] = clusters_intensity_sum[i][0] + image[clusters[i][j][0], clusters[i][j][1], 0]
                    clusters_intensity_sum[i][1] = clusters_intensity_sum[i][1] + image[clusters[i][j][0], clusters[i][j][1], 1]
                    clusters_intensity_sum[i][2] = clusters_intensity_sum[i][2] + image[clusters[i][j][0], clusters[i][j][1], 2]
                for c in range(channel):
                    cluster_intensity_average[i][c] = (clusters_intensity_sum[i][c] / len(clusters[i]))
                centroids[i] = cluster_intensity_average[i]
            updateI = updateI + 1
            print("update centroids! ", updateI)
            for i in range(k):
                print("No.", i, "cluster's centroid:", centroids[i])
            # calculate objective function J after update centroids
            Jsum_inside_clusters = np.zeros(k)
            for m in range(height):
                for n in range(width):
                    DistanceToCentroids = []
                    for i in range(k):
                        DistanceToCentroids.append(np.sqrt(int(image[m, n, 0] - centroids[i][0])**2 + int(image[m, n, 1] - centroids[i][1])**2 +int(image[m, n, 2] - centroids[i][2])**2))
                    for i in range(k):
                        if DistanceToCentroids.index(min(DistanceToCentroids)) == i:
                            Jsum_inside_clusters[i] = Jsum_inside_clusters[i] + DistanceToCentroids[i]
                        else:
                            pass
            Jsum_all_clusters_new = sum(Jsum_inside_clusters)
            J.append(Jsum_all_clusters_new)
            if abs(Jsum_all_clusters_old - Jsum_all_clusters_new) <= 1:
                break
            else:
                # flush clusters for assignment with new centroids
                clusters = []
                for i in range(k):
                    clusters.append([])

        plt.figure()
        plt.title(image_name + " RGB Objective function J")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.text(max(list(range(updateI))), max(J), "K = " + str(k))
        plt.plot(list(range(updateI)), J)
        path = os.getcwd()# get current path
        output_path = path + '/output/'
        plt.savefig(output_path + "RGB Loss for K=" + str(k) + " " + image_name + ".jpg")
        for i in range(k):
            for [index, coordinates] in enumerate(clusters[i]):
                image[coordinates[0], coordinates[1]] = centroids[i]
        cv2.imwrite(output_path + str(k) + 'RGB__After_Segmentation_' + image_name + ".jpg", image)
        return image
