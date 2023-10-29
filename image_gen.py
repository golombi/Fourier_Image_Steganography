import requests
import cv2
import numpy as np
from io import BytesIO
import time
from block_iterator import ImageBlockIterator

h_coef=125
w_coef=50



class BlockGen:
    def __init__(self, M, N, max_retries=10, retry_delay=4):
        self.M = M
        self.N = N
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.iter = None
        self.image_counter = 0
    def gen_block(self):
        if self.iter==None:
            self.image_counter+=1
            print("Using image number: " + str(self.image_counter))
            self.iter = ImageBlockIterator(get_image(h_coef*self.M, w_coef*self.N), self.M, self.N)
        try:
            for i in range(np.random.randint(h_coef*w_coef/5-3,h_coef*w_coef/5+3)):
                next(self.iter)
            return next(self.iter)
        except StopIteration:
            self.iter = None 
            return self.gen_block()
        
        


def get_image(M, N, max_retries=10, retry_delay=4):
    if max_retries == 0:
        print("Max retries reached. Failed to retrieve image.")
        return None
    
    try:
        # Send a GET request to the URL
        url = f"https://picsum.photos/{M}/{N}"
        response = requests.get(url)

        if response.status_code == 200:
            # Read the image from the response content
            image_bytes = BytesIO(response.content)
            image_data = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)

            # Decode and open the image using OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            # Display the image (you can remove this line if you don't want to display it)
            #cv2.imshow("Received Image", image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            return image
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
            time.sleep(retry_delay)  # Add a delay before retrying
            return get_image(M, N, max_retries=max_retries - 1, retry_delay=retry_delay)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        time.sleep(retry_delay)  # Add a delay before retrying
        return get_image(M, N, max_retries=max_retries - 1, retry_delay=retry_delay)

# Call the function to get and display the image
if __name__ == "__main__":
    get_image(11, 11)
