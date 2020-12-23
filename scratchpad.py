def tailgate_detect_and_mask(image):

    image_area = image.shape[0] * image.shape[1]
    mode_of_image = get_mode(image.copy())

    full_process = [f'mode of image: {mode_of_image}']

    if mode_of_image >= 125: # The mode of the image will determine if the image needs to be darkened or lightened
        first_range_tens = np.arange(0,-111, -10)
        full_process.append('darken process')
    else:
        first_range_tens = np.arange(0,111, 10)
        full_process.append('lighten process')

    second_range_tens = np.arange(0,111, 10)
   
    # Multiple sections of for loops used as edge detection should be primary,
    # Corner detection secondary
    # And border creation to force-close edges a last resort


    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75) #bilateral filter slower than gaussian but maintains edges better
            edges = layered_edge_detection(bilat.copy())
            sorted_contours = get_contours(edges)

            if len(sorted_contours) > 0:
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.75):
                    full_process.append('contour > 75% of area')

                    hull = cv2.convexHull(max_contour)

                    # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    print(f'full process [ {(" >>> ").join(full_process)} ]')

                    return masked_image

                elif cv2.contourArea(max_contour) < int(image_area*.75):
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.75):
                        full_process.append('contours added > 75% of area')

                        concat = np.concatenate((sorted_contours[0],sorted_contours[1]), axis=0)

                        hull = cv2.convexHull(concat)

                        # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_tailgate_mask(image, hull)
                        full_process.append('masked')

                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        # ax1.imshow(image)
                        # ax1.axis('off')
                        # ax2.imshow(masked_image)
                        # ax2.axis('off')
                        # plt.tight_layout();

                        print(f'full process [ {(" >>> ").join(full_process)} ]')

                        return masked_image

    full_process.append('edge processes tried')

    # Corner Processing         
    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75) #bilateral filter slower than gaussian but maintains edges better
            edges = layered_edge_detection(bilat.copy())
            sorted_contours = get_contours(edges)
            corners_array = get_array_of_corners(bilat.copy())

            if len(sorted_contours) > 0 and len(corners_array > 0):
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)
   

                max_contour_and_corners = np.concatenate((sorted_contours[0],np.array(corners_array)), axis=0)

                if cv2.contourArea(max_contour_and_corners) > int(image_area*.75):
                    full_process.append('corners + contour > 75% of area')

                    hull = cv2.convexHull(max_contour_and_corners)

                    # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    print(f'full process [ {(" >>> ").join(full_process)} ]')

                    return masked_image 

    full_process.append('corner processes tried')          

    # If all else fails.... border process
    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75)  

            # adding border around image to force-close contours as a last measure
            bordered_edges = border_process(bilat.copy())
            bordered_sorted_contours = get_contours(bordered_edges.copy())
            # drawn_ctrs = cv2.drawContours(bilat.copy(), bordered_sorted_contours, -1, (0, 255, 0), 2)
            # plt.imshow(drawn_ctrs)
               
            if len(bordered_sorted_contours) > 0:
                max_contour = bordered_sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.75):
                    full_process.append('bordered >>> contour > 75% of area')

                    hull = cv2.convexHull(max_contour)

                    # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    print(f'full process [ {(" >>> ").join(full_process)} ]')

                    return masked_image

                elif cv2.contourArea(max_contour) < int(image_area*.75):
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.75):
                        full_process.append('bordered >>> contours added > 75% of area')

                        concat = np.concatenate((sorted_contours[0],sorted_contours[1]), axis=0)

                        hull = cv2.convexHull(concat)

                        # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_tailgate_mask(image, hull)
                        full_process.append('masked')

                        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        # ax1.imshow(image)
                        # ax1.axis('off')
                        # ax2.imshow(masked_image)
                        # ax2.axis('off')
                        # plt.tight_layout();

                        print(f'full process [ {(" >>> ").join(full_process)} ]')

                        return masked_image

    full_process.append('border processes tried')                

    full_process.append('tailgate not found')
    print(f'full tailgate process [ {(" >>> ").join(full_process)} ]')
