from scipy.stats import mode
import opencv as cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mode(color_image):
    grayscale = cv2.cvtColor(color_image.copy(), cv2.COLOR_BGR2GRAY)
    mode_of_image, count = mode(grayscale, axis=None)
    return int(mode_of_image[0])


def adaptive_histogram(image):
    grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    cl1 = clahe.apply(grayscale)
    return cl1


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def auto_canny(image, sigma=0.66):
    #compute median of single channel pixel intentsities
    med = np.median(image)

    #apply automatic canny edge detection using median and sigma
    lower = int(max(0, (1.0-sigma)*med))
    upper = int(min(255, (1.0+sigma)*med))
    edged = cv2.Canny(image, lower, upper, apertureSize=3)

    return edged


def layered_edge_detection(bilat):
    edges1 = auto_canny(bilat.copy())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    dilated = cv2.dilate(edges1.copy(), kernel) #dilating edges to connect segments

    edges2 = auto_canny(dilated.copy())

    return edges2


def get_contours(edged_image):
    cnts, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)

    return sorted_contours


def add_border(truck_image):
    bordered_image = cv2.copyMakeBorder(truck_image, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,255,0))
    return bordered_image

def remove_border(bordered_image):
    cropped = bordered_image[2:bordered_image.shape[0]-2, 2:bordered_image.shape[1]-2] #necessary to crop border
    return cropped

def border_process(bilat_image):
    border = add_border(bilat_image)
    edges = layered_edge_detection(border)
    cropped = remove_border(edges.copy())

    return cropped


def transparent_tailgate_mask(orig_image, hull):
    BGRA = cv2.cvtColor(orig_image.copy(), cv2.COLOR_BGR2BGRA)
    masked = cv2.drawContours(BGRA.copy(), [hull], -1, (0,0,0,0), -1)

    masked_image = cv2.bitwise_and(BGRA, masked)
    return masked_image


def tailgate_detect_and_mask(image):

    image_area = image.shape[0] * image.shape[1]
    mode_of_image = get_mode(image.copy())

    full_process = [f'mode of image: {mode_of_image}']

    if mode_of_image > 125:
        first_range_tens = np.arange(0,-111, -10)
        full_process.append('darken process')
    elif mode_of_image < 125:
        first_range_tens = np.arange(0,111, 10)
        full_process.append('lighten process')

    second_range_tens = np.arange(0,111, 10)
   

    for i in first_range_tens:
        for j in second_range_tens:
            contrast = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(contrast.copy(),9,75,75)  #gaussian blur faster than bilateralFilter
            edges = layered_edge_detection(bilat.copy())
            sorted_contours = get_contours(edges)

            if len(sorted_contours) > 0:
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.75):
                    full_process.append('contour > 75% of area')

                    hull = cv2.convexHull(max_contour)

                    drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    print(f'full process [ {(" -> ").join(full_process)} ]')

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

                        print(f'full process [ {(" -> ").join(full_process)} ]')

                        return masked_image

                    else:
                        pass
                else:
                    pass

            
            # adding border around image to force-close contours as a last measure
            bordered_edges = border_process(bilat.copy())
            bordered_sorted_contours = get_contours(bordered_edges.copy())
            drawn_ctrs = cv2.drawContours(bilat.copy(), bordered_sorted_contours, -1, (0, 255, 0), 2)
            # plt.imshow(drawn_ctrs)
               
            if len(bordered_sorted_contours) > 0:
                max_contour = bordered_sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.75):
                    full_process.append('bordered -> contour > 75% of area')

                    hull = cv2.convexHull(max_contour)

                    drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_tailgate_mask(image, hull)
                    full_process.append('masked')

                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    # ax1.imshow(image)
                    # ax1.axis('off')
                    # ax2.imshow(masked_image)
                    # ax2.axis('off')
                    # plt.tight_layout();
                    
                    print(f'full process [ {(" -> ").join(full_process)} ]')

                    return masked_image

                elif cv2.contourArea(max_contour) < int(image_area*.75):
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.75):
                        full_process.append('bordered -> contours added > 75% of area')

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

                        print(f'full process [ {(" -> ").join(full_process)} ]')

                        return masked_image
                    
            else:
                pass
    full_process.append('tailgate not found')
    print(f'full tailgate process [ {(" -> ").join(full_process)} ]')


def handle_detect_and_mask(image):

    image_area = image.shape[0] * image.shape[1]
    mode_of_image = get_mode(image.copy())

    full_process = [f'mode of image: {mode_of_image}']

    if mode_of_image > 125:
        first_range_tens = np.arange(0,-111, -10)
        full_process.append('darken process')
    elif mode_of_image < 125:
        first_range_tens = np.arange(0,111, 10)
        full_process.append('lighten process')

    second_range_tens = np.arange(0,111, 10)
   

    for i in first_range_tens:
        for j in second_range_tens:
            
            #adapted = adaptive_histogram(image.copy())
            #bilat = cv2.bilateralFilter(image.copy(),9,75,75)
            adjusted = apply_brightness_contrast(image.copy(), brightness=i, contrast=j)
            bilat = cv2.bilateralFilter(adjusted.copy(),11,75,75)

            edges = layered_edge_detection(adjusted.copy())
            sorted_contours = get_contours(edges)

            if len(sorted_contours) > 0:
                max_contour = sorted_contours[0] #largest contour (hopefully the tailgate)

                if cv2.contourArea(max_contour) > int(image_area*.4):
                    print(f'area of max contour: {cv2.contourArea(max_contour)}')
                    print(f'.4 * image area: {int(image_area*.4)}')
                    print(f'image area: {image_area}')
                    full_process.append('contour > 40% of area')

                    hull = cv2.convexHull(max_contour)

                    drawn_ctrs = cv2.drawContours(adjusted.copy(), [hull], -1, (0, 255, 0), 2)

                    masked_image = transparent_handle_mask(image, hull)
                    full_process.append('masked')

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                    ax1.imshow(image)
                    ax1.axis('off')
                    ax2.imshow(masked_image)
                    ax2.axis('off')
                    plt.tight_layout();
                    
                    print(f'full process [ {(" -> ").join(full_process)} ]')

                    return masked_image

                elif cv2.contourArea(max_contour) < int(image_area*.40) \
                and len(sorted_contours) > 1:
                    if (cv2.contourArea(sorted_contours[0]) + 
                        cv2.contourArea(sorted_contours[1])) > int(image_area*.4):
                        full_process.append('contours added > 40% of area')

                        concat = np.concatenate((sorted_contours[0],sorted_contours[1]), axis=0)

                        hull = cv2.convexHull(concat)

                        # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_handle_mask(image, hull)
                        full_process.append('masked')

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        ax1.imshow(image)
                        ax1.axis('off')
                        ax2.imshow(masked_image)
                        ax2.axis('off')
                        plt.tight_layout();

                        print(f'full process [ {(" -> ").join(full_process)} ]')

                        return masked_image

                elif cv2.contourArea(max_contour) < int(image_area*.4):
                    sorted_contours2 = contours_from_edges_on_contours(image.copy(), sorted_contours)
                    
                    if cv2.contourArea(sorted_contours2[0]) > int(image_area*.4):
                        full_process.append('contour of contour > 40% of area')

                        hull = cv2.convexHull(sorted_contours2[0])

                        drawn_ctrs = cv2.drawContours(adjusted.copy(), [hull], -1, (0, 255, 0), 2)

                        masked_image = transparent_handle_mask(image, hull)
                        full_process.append('masked')

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                        ax1.imshow(image)
                        ax1.axis('off')
                        ax2.imshow(masked_image)
                        ax2.axis('off')
                        plt.tight_layout();
                        
                        print(f'full process [ {(" -> ").join(full_process)} ]')

                        return masked_image
                    
                    elif cv2.contourArea(sorted_contours2[0]) < int(image_area*.4) \
                    and len(sorted_contours2) > 1:
                        if (cv2.contourArea(sorted_contours2[0]) + 
                            cv2.contourArea(sorted_contours2[1])) > int(image_area*.4):
                            full_process.append('contour of contours added together > 40% of area')

                            concat = np.concatenate((sorted_contours2[0],sorted_contours2[1]), axis=0)

                            hull = cv2.convexHull(concat)

                            # drawn_ctrs = cv2.drawContours(contrast.copy(), [hull], -1, (0, 255, 0), 2)

                            masked_image = transparent_handle_mask(image, hull)
                            full_process.append('masked')

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
                            ax1.imshow(image)
                            ax1.axis('off')
                            ax2.imshow(masked_image)
                            ax2.axis('off')
                            plt.tight_layout();

                            print(f'full process [ {(" -> ").join(full_process)} ]')

                            return masked_image
                else:
                    pass
            else:
                pass
    print(f'brightness {i} | contrast {j}')
    full_process.append('handle not found')
    print(f'full handle process [ {(" -> ").join(full_process)} ]')
