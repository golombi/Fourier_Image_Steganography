import cv2

class ImageBlockIterator:
    def __init__(self, image, M, N):
        self.image = image
        self.M = M
        self.N = N
        self.row = 0
        self.col = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.row < self.image.shape[0]:
            block = self.image[self.row:self.row + self.N, self.col:self.col + self.M]
            self.col += self.M

            if self.col >= self.image.shape[1]:
                self.col = 0
                self.row += self.N

            return block
        else:
            raise StopIteration

if __name__ == "__main__":
    # Load an image
    image = cv2.imread('giraffe.jpg')

    # Specify the block dimensions (M, N)
    M = 200 # Width of the block
    N = 200  # Height of the block

    # Create an iterator for the image
    block_iterator = ImageBlockIterator(image, M, N)

    # Process and display each block
    for block in block_iterator:
        # Display the current block
        cv2.imshow("Block", block)
        
        # Perform your processing on the block here
        # 'block' contains the current block of the image
        # You can apply any image processing operation to 'block' in this loop
        
        # Wait for a key press to proceed to the next block
        cv2.waitKey(0)

    # Close the OpenCV window when done
    cv2.destroyAllWindows()
