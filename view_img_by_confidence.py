import os
import shutil

import cv2
import pandas as pd


def visualize_image(img_path, group_df):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    for _, row in group_df.iterrows():
        xc = float(row["x_center"])
        yc = float(row["y_center"])
        bw = float(row["width"])
        bh = float(row["height"])
        conf = float(row["confidence"])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    min_conf = group_df["confidence"].min()
    cv2.putText(
        img,
        f"min_conf: {min_conf:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Viewer (multi-face, sorted by conf)", img)


if __name__ == "__main__":
    autu_play = False

    input_dir = r'.\marked_results'
    output_dir = r'.\good_results'
    csv_path = r'.\marked_results\summary.csv'

    df = pd.read_csv(csv_path)
    df["image_key"] = df["relpath"]
    grouped = df.groupby("image_key")

    image_scores = grouped["confidence"].min().reset_index()
    image_scores = image_scores.sort_values("confidence", ascending=True)  # sort by confidence

    print(f"Total images: {len(image_scores)}")

    for _, row in image_scores.iterrows():
        relpath = row["image_key"]
        score = row["confidence"]

        img_path = os.path.join(input_dir, relpath)
        label_path = f'{os.path.dirname(img_path)}/{os.path.basename(img_path).split('.')[0]}.txt'
        group_df = grouped.get_group(relpath)

        print(f"Image: {relpath} | min_conf={score:.3f}")

        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        visualize_image(img_path, group_df)

        if autu_play:
            key = cv2.waitKey(500) & 0xFF  # ms
            if key == 13:  # enter key to skip
                pass
            elif key == 32:  # space key to pause
                while 1:
                    if cv2.waitKey(200) & 0xFF == 32:
                        break
            elif key == 27:  # ESC key to exit
                raise Exception
        else:
            while True:
                key = cv2.waitKey(20) & 0xFF  # ms
                if key == ord('0'):
                    dst_img_path = os.path.join(output_dir, os.path.relpath(img_path, input_dir))
                    dst_label_path = os.path.join(output_dir, os.path.relpath(label_path, input_dir))

                    os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                    shutil.copy(img_path, dst_img_path)
                    shutil.copy(label_path, dst_label_path)
                    print(f"Copied label to: {dst_label_path}, image to: {dst_img_path}")
                    break
                elif key == 13:  # enter key to skip this image
                    break
                elif key == 27:  # ESC key to exit
                    raise Exception

    cv2.destroyAllWindows()
