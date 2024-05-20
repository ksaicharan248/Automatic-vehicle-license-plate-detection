import cv2
from ultralytics import YOLO
import os
import PIL

match = ['0' , '1' , '9' , 'A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G' , 'H' , 'I' , '2' , 'J' , 'K' , 'L' , 'M' , 'N' ,
         'O' , 'P' , 'Q' , 'R' , '2' , 'S' , 'T' , 'U' , 'V' , 'W' , 'X' , 'Y' , 'Z' , '38' , '3' , '4' , '5' , '6' ,
         '7' , '8']

ocr_model = YOLO(r'./models/textOcr.pt')
lpmodel = YOLO(r"./models/best25.pt")


def sort_coordinates(combined) :
    textid = {0 : '0' , 1 : '1' , 2 : '2' , 3 : '3' , 4 : '4' , 5 : '5' , 6 : '6' , 7 : '7' , 8 : '8' , 9 : '9' ,
              10 : 'A' , 11 : 'B' , 12 : 'C' , 13 : 'D' , 14 : 'E' , 15 : 'F' , 16 : 'G' , 17 : 'H' , 18 : 'I' ,
              19 : 'J' , 20 : 'K' , 21 : 'L' , 22 : 'M' , 23 : 'N' , 24 : 'O' , 25 : 'P' , 26 : 'Q' , 27 : 'R' ,
              28 : 'S' , 29 : 'T' , 30 : 'U' , 31 : 'V' , 32 : 'W' , 33 : 'X' , 34 : 'Y' , 35 : 'Z'}

    sorted_by_y = sorted(combined , key=lambda x : x[1])

    if len(sorted_by_y) > 0 :
        avg_y = (sum(coord[1] for coord in sorted_by_y) / len(sorted_by_y)) + 20
    else :
        avg_y = sorted_by_y[len(sorted_by_y) - 1]
    score = sum(coord[4] for coord in sorted_by_y) / len(sorted_by_y)
    row1 = [coord for coord in sorted_by_y if coord[1] < avg_y]
    row2 = [coord for coord in sorted_by_y if coord[1] >= avg_y]
    sorted_row1 = sorted(row1 , key=lambda x : x[0])
    sorted_row2 = sorted(row2 , key=lambda x : x[0])
    text = ''
    for i in sorted_row1 :
        text += i[5]
    for i in sorted_row2 :
        text += i[5]
    return [text , score]

def detect_from_image(image):
    #img = cv2.imread(os.path.join(path , image))
    img = image
    # cv2.imshow('img' , img)
    # img = img.resize((640 , 640))
    lpresult = lpmodel.predict(source=img , conf=0.4)
    for lpres in lpresult[0].boxes.data.tolist() :
        lx1 , ly1 , lx2 , ly2 , lscore , lclass_id = lpres
        # crop the license plate
        license_plate_crop = img[int(ly1) :int(ly2) , int(lx1) :int(lx2)]
        # cv2.imshow('license plate' , license_plate_crop)
        out = []
        ocrRes = ocr_model.predict(source=license_plate_crop , conf=0.4)
        for ocrResult in ocrRes[0].boxes.data.tolist() :
            ox1 , oy1 , ox2 , oy2 , oscore , oclass_id = ocrResult
            #cv2.rectangle(license_plate_crop , (int(ox1) , int(oy1)) , (int(ox2) , int(oy2)) , (0 , 0 , 255) , 1)
            cv2.putText(license_plate_crop , f'{match[int(oclass_id)]}' , (int(ox1) , int(oy1)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,
                        (0 , 255 , 0) , 1 , cv2.LINE_AA)
            out.append([ox1 , oy1 , ox2 , oy2 , oscore , match[int(oclass_id)]])
        license_plate_text , _ = sort_coordinates(out)
        cv2.rectangle(img , (int(lx1) , int(ly1)) , (int(lx2) , int(ly2)) , (0 , 0 , 255) , 1)
        cv2.putText(img , f'{license_plate_text}' , (int(lx1) , int(ly1 - 10)) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 ,
                    (0 , 255 , 0) , 1 , cv2.LINE_AA)

        return img , license_plate_text , license_plate_crop
if "__main__" == __name__ :
    sup = 1

    if sup == 1 :
        try :
            path = r'./localimages'
            for image in os.listdir(path) :
                img = cv2.imread(os.path.join(path , image))
                #cv2.imshow('img' , img)
                # img = img.resize((640 , 640))
                lpresult = lpmodel.predict(source=img , conf=0.4)
                for lpres in lpresult[0].boxes.data.tolist() :
                    lx1 , ly1 , lx2 , ly2 , lscore , lclass_id = lpres
                    # crop the license plate
                    license_plate_crop = img[int(ly1) :int(ly2) , int(lx1) :int(lx2)]
                    cv2.imwrite('./crop1/'+image , license_plate_crop)
                    #cv2.imshow('license plate' , license_plate_crop)
                    out = []
                    ocrRes = ocr_model.predict(source=license_plate_crop , conf=0.4)
                    for ocrResult in ocrRes[0].boxes.data.tolist() :
                        ox1 , oy1 , ox2 , oy2 , oscore , oclass_id = ocrResult
                        cv2.rectangle(license_plate_crop , (int(ox1) , int(oy1)) , (int(ox2) , int(oy2)) , (0 , 0 , 255) , 1)
                        cv2.putText(license_plate_crop , f'{match[int(oclass_id)]}' , (int(ox1) , int(oy1)-3) ,
                                    cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255 , 0) , 5 , cv2.LINE_AA)
                        cv2.imwrite('./crop/' + image , license_plate_crop)
                        out.append([ox1 , oy1 , ox2 , oy2 , oscore , match[int(oclass_id)]])
                    license_plate_text , _ = sort_coordinates(out)
                    cv2.rectangle(img , (int(lx1) , int(ly1)) , (int(lx2) , int(ly2)) , (0 , 0 , 255) ,
                                  10)
                    #cv2.putText(img , f'{license_plate_text}' , (int(lx1) , int(ly1 - 10)) , cv2.FONT_HERSHEY_SIMPLEX ,0.5 , (0 , 255 , 0) , 1 , cv2.LINE_AA)

                    cv2.imwrite('./store/' + image , img)
        except Exception as e :
            print(e)
            pass  # print(results)

    else :
        try :
            # path = r"C:\Users\Sai charan\vs_code_progaming_files\lpr.yolov8\test\images"
            pathimage = r'./datasets/img_1.png'
            while pathimage :  # for image in os.listdir(path) :   #
                img = cv2.imread(os.path.join(pathimage))
                cv2.imshow('img' , img)
                lpresult = lpmodel.predict(source=img , conf=0.4)
                for lpres in lpresult[0].boxes.data.tolist() :
                    lx1 , ly1 , lx2 , ly2 , lscore , lclass_id = lpres
                    license_plate_crop = img[int(ly1) :int(ly2) , int(lx1) :int(lx2)]
                    cv2.imshow('license plate' , license_plate_crop)
                    out = []
                    license_plate_crop = cv2.resize(license_plate_crop , (640 , 640))
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop , cv2.COLOR_BGR2GRAY)
                _ , license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray , 64 , 255 ,
                                                              cv2.THRESH_BINARY_INV)
                cv2.imshow('license plate' , license_plate_crop_thresh)
                ocrRes = ocr_model(license_plate_crop)
                for ocrResult in ocrRes[0].boxes.data.tolist() :
                    ox1 , oy1 , ox2 , oy2 , oscore , oclass_id = ocrResult
                    out.append([ox1 , oy1 , ox2 , oy2 , oscore , match[int(oclass_id)]])
                    cv2.rectangle(license_plate_crop , (int(ox1) , int(oy1)) , (int(ox2) , int(oy2)) , (0 , 0 , 255) ,
                                  1)
                    cv2.putText(license_plate_crop , f'{match[int(oclass_id)]}' , (int(ox1) , int(oy1 - 10)) ,
                                cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 255) , 1 , cv2.LINE_AA)
                try :
                    print(sort_coordinates(out))
                except :
                    pass

                cv2.imshow('ocr' , license_plate_crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e :
            print(e)
            pass  # print(results)
