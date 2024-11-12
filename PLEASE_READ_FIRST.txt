Description:
This program takes as input an image (png or jpg) that contains a set of unsolved jigsaw puzzle pieces. It solves the puzzle and displays the result in an output image.

Installation:
   1. Open project in Pycharm
   2. Validate that project include jigsaw.py, stanford-campus.png, and stanford-football.png are there.
   3. Run >python3 -m pip install Pillow
   4. Run >python3 -m pip install -U scikit-image
   5. Run >python3 jigsaw.py (stanford-campus.png or stanford-football.png)

Details:
This program uses the skills learned in CS106A - specifically around problem decomposition, image manipulation, data structures, and so much more! The complex
puzzle solving algorithm is made up of the following main parts:

- Read input image and determine the number of rows and columns that contain jigsaw puzzle pieces. The implementation uses the pillow library that we learned in class
for image processing and analysis.

- Extract each puzzle piece from the grid. Each puzzle piece under goes classification to determine each of its side types (edge, outie, innie).

- The matching algorithm uses a brute-force technique that compares each puzzle piece side with every other puzzle piece side - e.g. piece[m].side[n] X piece[m].side[n].
First a check is performed to see if the pieces are compatible - e.g. outie side with innie side. If compatible, a pixel sampling is taken across both sides of the
border where the pieces connect. The difference in the RGB values of the pixels are used to compute a compatibility score. The lower the difference, the higher
probability of a match.

- The solving algorithm finds the first corner puzzle piece and starts solving the puzzle left-to-right row-by-row. It finds the best puzzle piece match based on
compatibility scores computed in previous step. Results are stored in a grid.

- Now the puzzle has been solved, a final image is generated. It takes the solved puzzle grid and merges each piece to generate the output image.


Conclusion:
I’m super excited to submit my Jigsaw Puzzle Solver for the 106A Programming Contest! I thoroughly enjoyed the challenge!! I have pretty much spent every
free moment (plus some long nights), since the announcement of the contest working on it.  My mom always has a puzzle going at home, so to make things easier
for her, I thought I would try programming a jigsaw puzzle solver...and it actually works!!  I did come to the realization during the process, that this was a
pretty complex problem. It would have been better solved using AI, but nonetheless it was incredibly fun programming it with all of the stuff we learned in
class! I completely underestimated the complexity of the algorithms, and the amount of time it would take to program it - I wish I would have had more
time, so I could clean up my code a bit and I would have loved to make some more puzzles (it’s surprisingly hard to find custom puzzles online that you can cut
out) - but I’m excited for how it turned out all in all. My next venture is definitely going to be a beginner AI course - I would love to try again using AI!

Author:
Tannen Hall - rising high school Junior at Pacific Ridge High School in San Diego, CA
