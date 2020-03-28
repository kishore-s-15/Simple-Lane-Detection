# Simple Lane Detection using Python

Input image is first converted into *Grayscale*, smoothened using *Gaussian Blurring*, then *Canny Edge Detection* is applied to it to detect the edges.

The background noises and data such as sky, etc. are removed using *masking*.

The Lane lines in the input images are found using *Hough Lines Transform in Polar Co - Ordinates*, Polar Co - Ordinates is chosen because simple Hough Line Transform cannot detect perfectly horizontal and vertical lines.

There are many lines which are obtained from the previous step, which is smoothened out by *averaging* the left and right lane lines.

Then the lane lines are *drawn* on the input image, and using *weighted addition* of images, the input image and lane image is blended into a single output image.
