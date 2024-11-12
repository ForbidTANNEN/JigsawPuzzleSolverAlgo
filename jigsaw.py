import copy
from PIL import Image, ImageOps
from skimage import io
from scipy import ndimage
from skimage import measure
import sys
import time

"""
**PLEASE SEE 'PLEASE_READ_FIRST.txt'**

This  program takes as input an image (png or jpg) that contains a set of 
unsolved puzzle pieces. It solves the puzzle and displays the result in 
an output image.

Usage:
>python3 jigsaw.py <input image file>
"""

MIN_ROW_INDEX = 0
MAX_ROW_INDEX = 2

X = 0
Y = 1
RGBA = 3


class PuzzleSide:
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    LEFT = 4


PUZZLE_SIDES = [PuzzleSide.RIGHT, PuzzleSide.LEFT, PuzzleSide.TOP, PuzzleSide.BOTTOM]


class PuzzleSideType:
    OUTIE = 1
    INNIE = 2
    EDGE = 3


class PuzzleDimensions:
    def __init__(self, row_count, col_count):
        self.row_count = row_count
        self.col_count = col_count

    def __str__(self):
        return str("rows=" + str(self.row_count) + ",cols=" + str(self.col_count))


class PuzzlePiece:
    """
    Contains details about each puzzle piece including its image and each side type
    """
    def __init__(self, image, row, col, index):
        self.image = image
        self.width = 0
        self.row = row
        self.col = col
        self.index = index
        self.sides = {}
        self.num_of_rotations = 0

    def __str__(self):
        return str(self.index)


class PuzzleMatch:
    """
    Captures a compatibility score between two puzzle pieces and sides
    """
    def __init__(self, p1, p1_side, p2, p2_side, score):
        self.p1 = p1
        self.p1_side = p1_side
        self.p2 = p2
        self.p2_side = p2_side
        self.score = score

    def __str__(self):
        return str(self.p1.index) + ":" + str(self.p1_side) + "->" + str(self.p2.index) + ":" + str(self.p2_side) + "=" + str(self.score)


class PuzzleMatchSide:
    """
    Captures a relationship between a puzzle piece and a side
    """
    def __init__(self, piece, side):
        self.piece = piece
        self.side = side


def get_puzzle_regions(image):
    """
    Given an input image, returns how many puzzle pieces are present in the PNG image
    """
    im = io.imread(image, as_gray=True)
    puzzle = ndimage.binary_fill_holes(im < 1)
    # plt.imshow(puzzle, cmap='gray')
    # plt.show()
    return measure.label(puzzle)


def get_puzzle_dimensions(image):
    """
    Given an input PNG, returns how many puzzle pieces are in each row as well as how many puzzle pieces are in each column.
    Returns a tuple
    """
    puzzle_regions = get_puzzle_regions(image)
    puzzle_region_props = measure.regionprops(puzzle_regions)
    puzzle_pieces_total = puzzle_regions.max()
    average_y = 0
    puzzle_pieces_col = 0
    first_label = True
    for label_prop in puzzle_region_props:
        bbox = label_prop.bbox
        if first_label:
            first_label = False
            average_y = (bbox[MIN_ROW_INDEX] + bbox[MAX_ROW_INDEX])/2
        elif not (bbox[MAX_ROW_INDEX] >= average_y >= bbox[MIN_ROW_INDEX]):
            break
        puzzle_pieces_col += 1
    puzzle_pieces_row = puzzle_pieces_total // puzzle_pieces_col

    return PuzzleDimensions(puzzle_pieces_row, puzzle_pieces_col)


def get_puzzle_pieces(input_image, puzzle_dims):
    """
    Takes in an input image, a puzzle row integer count, and a puzzle column integer count.
    Returns a list of all the seperated puzzle piece objects
    """
    image = Image.open(input_image)
    puzzle_pieces = []

    puzzle_capture_width = image.width // puzzle_dims.col_count
    puzzle_capture_height = image.height // puzzle_dims.row_count
    count = 0
    # row_index = 2
    # col_index = 4
    for row_index in range(puzzle_dims.row_count):
        for col_index in range(puzzle_dims.col_count):
            count += 1
            x_offset = col_index * puzzle_capture_width
            y_offset = row_index * puzzle_capture_width
            area = (x_offset, y_offset, x_offset + puzzle_capture_width, y_offset + puzzle_capture_height)
            cropped_image = image.crop(area)
            # cropped_image.show()
            puzzle_piece = PuzzlePiece(cropped_image, row_index, col_index, count - 1)
            set_puzzle_piece_attributes(puzzle_piece)
            puzzle_pieces.append(puzzle_piece)

    return puzzle_pieces


def get_puzzle_piece_width(puzzle_piece):
    """
    gets the width of a puzzle piece (e.g. not transparent pixels)
    """
    # result = {}
    center = puzzle_piece.image.height // 2
    puzzle_piece_length = 0
    first = True
    for x in range(puzzle_piece.image.width):
        pixel = puzzle_piece.image.getpixel((x, center))
        if pixel[RGBA] != 0:
            if first:
                # first_cord = (x, center)
                first = False
            puzzle_piece_length += 1
    # result['puzzle_piece_length'] = puzzle_piece_length
    # result['first_cord'] = first_cord
    return puzzle_piece_length


def get_first_coord(puzzle_piece):
    """
    returns the frist coordinate of the non-transparent pixel in middle of puzzle piece
    """
    center = puzzle_piece.image.height // 2
    first = True
    first_cord = ()
    for x in range(puzzle_piece.image.width):
        pixel = puzzle_piece.image.getpixel((x, center))
        if pixel[RGBA] != 0:
            if first:
                first_cord = (x, center)

                first = False
    return first_cord


def connect_pieces(main_image, piece_one, place_cords, first_cord):
    """
    merges two puzzle pieces
    """
    for y in range(piece_one.image.height):
        for x in range(piece_one.image.width - first_cord[X]):
            pixel = piece_one.image.getpixel((x + first_cord[X], y))
            if pixel[RGBA] != 0:
                main_image.putpixel((place_cords[X] + x, place_cords[Y] + y), pixel)
        for x in range(first_cord[X]):
            pixel = piece_one.image.getpixel((first_cord[X] - x, y))
            if pixel[RGBA] != 0:
                main_image.putpixel((place_cords[X] - x, place_cords[Y] + y), pixel)
    return main_image


def get_three_x_values(puzzle_piece, side, pixel_grabs):
    """
    Given puzzle piece object, returns three x values of the right side. The top, middle, and bottom
    """
    puzzle_image = copy.deepcopy(puzzle_piece.image)

    if side == 3:
        puzzle_image = puzzle_image.rotate(90)
    elif side == 4:
        puzzle_image = puzzle_image.rotate(180)
    elif side == 1:
        puzzle_image = puzzle_image.rotate(270)
    #vertically flips image
    elif side == 5:
        puzzle_image = ImageOps.mirror(puzzle_image)

    # puzzle_image.show()

    half_width = round((puzzle_image.width // 2) * 1.3)

    half_height = puzzle_image.height // 2
    center = (half_width, half_height)
    full_offset = round(puzzle_image.width * 0.23)
    offset = round(full_offset - (puzzle_image.width * 0.1))/pixel_grabs

    bottom_edge = []
    top_edge = []

    middle_edge = ()
    middle_first = False

    for x in range(puzzle_image.width // 2):
        x_width = x + (puzzle_image.width // 2)
        cords = (x_width, half_height)
        pixel = puzzle_image.getpixel(cords)
        if middle_first == False and pixel[RGBA] == 0:
            middle_edge = cords
            middle_first = True
            puzzle_image.putpixel(cords, (255, 0, 0))
    for i in range(pixel_grabs):
        top_first = False
        bottom_first = False
        i = i+1
        for x in range(puzzle_image.width - half_width):
            x_width = x + half_width
            cords = (x_width, (half_height + full_offset) - round(offset*i))
            pixel = puzzle_image.getpixel(cords)
            if top_first == False and pixel[RGBA] == 0:
                top_edge.append(cords)
                top_first = True
                puzzle_image.putpixel(cords, (255, 0, 0))

            cords = (x_width, (half_height - full_offset) + round(offset*i))
            pixel = puzzle_image.getpixel(cords)
            if bottom_first == False and pixel[RGBA] == 0:
                bottom_edge.append(cords)
                bottom_first = True
                puzzle_image.putpixel(cords, (255, 0, 0))
                # print(top_edge)
                # print(middle_edge)
                # print(bottom_edge)
    # puzzle_image.show()
    return ({'top_edge': top_edge[0], 'middle_edge': middle_edge, 'bottom_edge': bottom_edge[0], 'returned_image': puzzle_image, 'top_list': top_edge, 'bottom_list': bottom_edge})


def check_edge_type(puzzle_piece, side):
    """
    Given a puzzle piece, function returns whether the right edge is a outie, innie, or edge
    """
    edges = get_three_x_values(puzzle_piece, side, 3)

    top_edge = edges['top_edge'][X]
    middle_edge = edges['middle_edge'][X]
    bottom_edge = edges['bottom_edge'][X]

    buffer = round(puzzle_piece.width * 0.02)

    if top_edge - buffer < middle_edge < top_edge + buffer or bottom_edge - buffer < middle_edge < bottom_edge + buffer:
        return PuzzleSideType.EDGE
    # innie
    elif middle_edge < bottom_edge + buffer or middle_edge < top_edge + buffer:
        return PuzzleSideType.INNIE

    # outie
    elif middle_edge > bottom_edge - buffer or middle_edge > bottom_edge - buffer:
        return PuzzleSideType.OUTIE



def set_puzzle_piece_attributes(puzzle_piece):
    """
    given a puzzle piece, compute and set
    its attributes including its side types
    """
    puzzle_piece.width = get_puzzle_piece_width(puzzle_piece)
    for side in PUZZLE_SIDES:
        puzzle_piece.sides[side] = check_edge_type(puzzle_piece, side)


def rotate_ninety_degrees(puzzle_piece):
    """
    rotate puzzle piece 90 degress clockwise
    """
    rotated_piece = copy.deepcopy(puzzle_piece)
    rotated_piece.image = rotated_piece.image.rotate(-90)

    rotated_piece.sides[PuzzleSide.TOP] = puzzle_piece.sides[PuzzleSide.LEFT]
    rotated_piece.sides[PuzzleSide.RIGHT] = puzzle_piece.sides[PuzzleSide.TOP]
    rotated_piece.sides[PuzzleSide.BOTTOM] = puzzle_piece.sides[PuzzleSide.RIGHT]
    rotated_piece.sides[PuzzleSide.LEFT] = puzzle_piece.sides[PuzzleSide.BOTTOM]

    return rotated_piece


def rotate_piece(puzzle_piece, from_side, to_side):
    rotated_piece = puzzle_piece
    num_of_rotations = 0
    if from_side < to_side:
        num_of_rotations = to_side - from_side
    elif to_side < from_side:
        num_of_rotations = 4 - abs(to_side - from_side)
    if num_of_rotations > 0:
        for rotate_num in range(num_of_rotations):
            rotated_piece = rotate_ninety_degrees(rotated_piece)
    rotated_piece.num_of_rotations = num_of_rotations
    return rotated_piece


def get_match_key(p, p_side):
    """creates dictionary key"""
    return str(p.index) + ":" + str(p_side)


def add_match(matches, p1, p1_side, p2, p2_side, score):
    """
    add compatibility score of two puzzle pieces and sides
    """
    match = PuzzleMatch(p1, p1_side, p2, p2_side, score)
    match_key = get_match_key(p1, p1_side)
    match_list = []
    if match_key in matches:
        match_list = matches[match_key]
    match_list.append(match)
    matches[match_key] = match_list

    match = PuzzleMatch(p2, p2_side, p1, p1_side, score)
    match_key = get_match_key(p2, p2_side)
    match_list = []
    if match_key in matches:
        match_list = matches[match_key]
    match_list.append(match)
    matches[match_key] = match_list


def del_match(matches, p1, p1_side, p2, p2_side):
    """
    removes a match, so can't be reused when solving puzzle
    """
    p1_match_key = get_match_key(p1, p1_side)
    p2_match_key = get_match_key(p2, p2_side)

    if p1_match_key in matches:
        del matches[p1_match_key]

    if p2_match_key in matches:
        del matches[p2_match_key]


def does_match_exist(matches, p1, p1_side, p2, p2_side):

    if p2.index < p1.index:
        temp_p = p2
        temp_s = p2_side
        p2 = p1
        p2_side = p1_side
        p1 = temp_p
        p1_side = temp_s

    match_key = get_match_key(p1, p1_side)
    if match_key in matches:
        for match in matches[match_key]:
            if match.p2.index == p2.index and match.p2_side == p2_side:
                return True

    return False


def get_score(p1, p2, image_file='image_file'):
    """
    computes a compatibility score between two pieces. it is based
    on comparing the RBG pixel values from a sampling of pixels along the
    matching puzzle piece edges
    """
    num_pixel_samples = 50
    if p1.image.width > 300:
        offset = round(p1.image.width * 0.005)
    else:
        offset = round(p1.image.width * 0.02)

    p1_three_cords = get_three_x_values(p1, PuzzleSide.RIGHT, num_pixel_samples)
    p2_three_cords = get_three_x_values(p2, 5, num_pixel_samples)

    # p1_three_cords['returned_image'].show()

    top_diff = 0
    p1_top_list = p1_three_cords['top_list']
    p2_top_list = p2_three_cords['top_list']
    for i in range(len(p1_top_list)):
        p1_top_pixel = p1.image.getpixel((p1_top_list[i][X] - offset, p1_top_list[i][Y]))
        p2_top_pixel = p2_three_cords["returned_image"].getpixel((p2_top_list[i][X] - offset, p2_top_list[i][Y]))
        top_diff += (abs(p1_top_pixel[0] - p2_top_pixel[0])) + (abs(p1_top_pixel[1] - p2_top_pixel[1])) + (abs(p1_top_pixel[2] - p2_top_pixel[2])) + (abs(p1_top_pixel[3] - p2_top_pixel[3]))

    bottom_diff = 0
    p1_bottom_list = p1_three_cords['bottom_list']
    p2_bottom_list = p2_three_cords['bottom_list']
    for i in range(len(p1_bottom_list)):
        p1_bottom_pixel = p1.image.getpixel((p1_bottom_list[i][X] - offset, p1_bottom_list[i][Y]))
        p2_bottom_pixel = p2_three_cords["returned_image"].getpixel((p2_bottom_list[i][X] - offset, p2_bottom_list[i][Y]))
        bottom_diff += (abs(p1_bottom_pixel[0] - p2_bottom_pixel[0])) + (abs(p1_bottom_pixel[1] - p2_bottom_pixel[1])) + (abs(p1_bottom_pixel[2] - p2_bottom_pixel[2])) + (abs(p1_bottom_pixel[3] - p2_bottom_pixel[3]))

    middle_diff = 0
    p1_middle_pixel = p1.image.getpixel((p1_three_cords["middle_edge"][X] - offset, p1_three_cords["middle_edge"][Y]))
    p2_middle_pixel = p2_three_cords["returned_image"].getpixel((p2_three_cords["middle_edge"][X] - offset, p2_three_cords["middle_edge"][Y]))
    middle_diff += (abs(p1_middle_pixel[0] - p2_middle_pixel[0])) + (abs(p1_middle_pixel[1] - p2_middle_pixel[1])) + (abs(p1_middle_pixel[2] - p2_middle_pixel[2])) + (abs(p1_middle_pixel[3] - p2_middle_pixel[3]))

    #final_diff = top_diff + bottom_diff + (middle_diff)

    final_diff = int(top_diff/num_pixel_samples) + int(bottom_diff/num_pixel_samples) + middle_diff

    return final_diff


def are_puzzle_pieces_compatible(p1, p2):
    """
    returns true if two pieces are compatible (irrespective of color match)
    """
    if p1.index != p2.index:
        if (p1.sides[PuzzleSide.RIGHT] == PuzzleSideType.OUTIE and p2.sides[PuzzleSide.LEFT] == PuzzleSideType.INNIE) or (p1.sides[PuzzleSide.RIGHT] == PuzzleSideType.INNIE and p2.sides[PuzzleSide.LEFT] == PuzzleSideType.OUTIE):
            if (p1.sides[PuzzleSide.TOP] == PuzzleSideType.EDGE and p2.sides[PuzzleSide.TOP] == PuzzleSideType.EDGE) or (p1.sides[PuzzleSide.TOP] != PuzzleSideType.EDGE and p2.sides[PuzzleSide.TOP] != PuzzleSideType.EDGE):
                if (p1.sides[PuzzleSide.BOTTOM] == PuzzleSideType.EDGE and p2.sides[PuzzleSide.BOTTOM] == PuzzleSideType.EDGE) or (p1.sides[PuzzleSide.BOTTOM] != PuzzleSideType.EDGE and p2.sides[PuzzleSide.BOTTOM] != PuzzleSideType.EDGE):
                    return True
    return False


def get_corner(puzzle_pieces):
    """
    returns the first corner
    """
    # returns the first corner puzzle piece
    # returns the first corner puzzle piece
    for p in puzzle_pieces:
        if (p.sides[PuzzleSide.LEFT] == PuzzleSideType.EDGE and p.sides[PuzzleSide.TOP] == PuzzleSideType.EDGE) or (p.sides[PuzzleSide.TOP] == PuzzleSideType.EDGE and p.sides[PuzzleSide.RIGHT] == PuzzleSideType.EDGE) or (p.sides[PuzzleSide.RIGHT] == PuzzleSideType.EDGE and p.sides[PuzzleSide.BOTTOM] == PuzzleSideType.EDGE) or (p.sides[PuzzleSide.BOTTOM] == PuzzleSideType.EDGE and p.sides[PuzzleSide.LEFT] == PuzzleSideType.EDGE):
            return p
    return None


def get_puzzle_piece_matches(puzzle_pieces):
    """
    brute force way of computing a compatibility score between
    every puzzle piece and edge
    """
    # this function returns a dictionary of all scores
    # of all compatible puzzle pieces
    matches = {}

    piece_count = len(puzzle_pieces)
    for p1 in puzzle_pieces:
        for s1 in PUZZLE_SIDES:
            for p2 in puzzle_pieces:
                for s2 in PUZZLE_SIDES:
                    if not does_match_exist(matches, p1, s1, p2, s2):
                        rp1 = rotate_piece(p1, s1, PuzzleSide.RIGHT)
                        rp2 = rotate_piece(p2, s2, PuzzleSide.LEFT)
                        if are_puzzle_pieces_compatible(rp1, rp2):
                            score = get_score(rp1, rp2)
                            add_match(matches, p1, s1, p2, s2, score)
                        else:
                            None
                    else:
                        None

    return matches


def rotate_corner(corner):
    """
    rotate puzzle piece to top-left corner
    """
    is_top_left = False
    rotated_corner = copy.deepcopy(corner)
    while not is_top_left:
        if rotated_corner.sides[PuzzleSide.LEFT] == PuzzleSideType.EDGE and rotated_corner.sides[PuzzleSide.TOP] == PuzzleSideType.EDGE:
            is_top_left = True
        else:
            rotated_corner = rotate_ninety_degrees(rotated_corner)
            rotated_corner.num_of_rotations = rotated_corner.num_of_rotations + 1
    return rotated_corner


def get_match(piece, side, matches, used_matches):
    """
    returns puzzle piece with best compatibility score
    of the input piece and side
    """

    match_key = get_match_key(piece, side)

    best_match = None
    best_score = 255*255*255*255*255 #max score

    if match_key in matches:
        match_list = matches[match_key]
        for item in match_list:
            if item.p2.index == piece.index:
                if item.p1.index not in used_matches:
                    if item.score < best_score:
                        best_match = item
                        best_score = item.score
            else:
                if item.p2.index not in used_matches:
                    if item.score < best_score:
                        best_match = item
                        best_score = item.score

        if best_match.p2.index == piece.index:
            return PuzzleMatchSide(best_match.p1, best_match.p1_side)
        else:
            return PuzzleMatchSide(best_match.p2, best_match.p2_side)
    return None


def get_match_side(rotated_puzzle_piece, side):
    rotations = rotated_puzzle_piece.num_of_rotations
    current_side = side
    while rotations > 0:
        current_side = get_counterclockwise_side(current_side)
        rotations = rotations - 1
    return current_side


def get_counterclockwise_side(current_side):
    if current_side == PuzzleSide.RIGHT:
        return PuzzleSide.TOP
    elif current_side == PuzzleSide.TOP:
        return PuzzleSide.LEFT
    elif current_side == PuzzleSide.LEFT:
        return PuzzleSide.BOTTOM
    elif current_side == PuzzleSide.BOTTOM:
        return PuzzleSide.RIGHT


def move_puzzle_pieces(puzzle_dimensions, puzzle_pieces, matches):
    """
    finds top-left corner piece and solves puzzle left-to-right and row-by-row
    by finding each piece based on best compatibility score
    """
    grid = {}

    used_matches = set()

    match_piece = get_corner(puzzle_pieces)
    for row_index in range(puzzle_dimensions.row_count):
        for col_index in range(puzzle_dimensions.col_count):
            if row_index == 0:
                if col_index == 0:
                    rotated_match_piece = rotate_corner(match_piece)
                    used_matches.add(match_piece.index)
                else:
                    left_piece = grid[row_index, col_index - 1]
                    left_piece_match_side = get_match_side(left_piece, PuzzleSide.RIGHT)
                    match_piece = get_match(left_piece, left_piece_match_side, matches, used_matches)
                    rotated_match_piece = rotate_piece(match_piece.piece, match_piece.side, PuzzleSide.LEFT)
                    del_match(matches, left_piece, left_piece_match_side, match_piece.piece, match_piece.side)
                    used_matches.add(match_piece.piece.index)
            else:
                top_piece = grid[row_index - 1, col_index]
                top_piece_match_side = get_match_side(top_piece, PuzzleSide.BOTTOM)
                match_piece = get_match(top_piece, top_piece_match_side, matches, used_matches)
                if match_piece is not None:
                    rotated_match_piece = rotate_piece(match_piece.piece, match_piece.side, PuzzleSide.TOP)
                    del_match(matches, top_piece, top_piece_match_side, match_piece.piece, match_piece.side)
                    if col_index > 0:
                        left_piece = grid[row_index, col_index - 1]
                        if left_piece is not None:
                            left_piece_match_side = get_match_side(left_piece, PuzzleSide.RIGHT)
                            del_match(matches, left_piece, left_piece_match_side, match_piece.piece, match_piece.side)
                    used_matches.add(match_piece.piece.index)
                else:
                    rotated_match_piece = None

            grid[row_index, col_index] = rotated_match_piece

    return grid


def generate_output_image(puzzle_dimensions, grid):
    """
    takes as input a grid. Each cell is a rotated puzzle piece.
    outputs an image of the resulting puzzle
    """

    width = grid[0,0].image.width
    height = grid[0,0].image.height
    out_image = Image.new('RGBA', (width * (puzzle_dimensions.col_count - 1), height * (puzzle_dimensions.row_count - 1)), (105,105,105))

    half_height = ((get_three_x_values(grid[0,0], PuzzleSide.TOP, 2)['middle_edge'][X]) - (get_first_coord(grid[0,0])[X]))/2

    for row_index in range(puzzle_dimensions.row_count):
        x_cord = 0
        for col_index in range(puzzle_dimensions.col_count):
            piece = grid[row_index, col_index]
            first_cord = get_first_coord(piece)
            piece_width = get_three_x_values(piece, PuzzleSide.RIGHT, 3)['middle_edge']
            out_image = connect_pieces(out_image, piece, (x_cord, round(half_height*2*row_index)), first_cord)
            x_cord += piece_width[X] - first_cord[X]
            out_image.show()
            time.sleep(0.7)


def print_matches(matches):
    for match_key in matches.keys():
        match_list = matches[match_key]
        for match in match_list:
            print(match)


def solve_puzzle(puzzle_in_image):
    """
    function that performs all the steps necessary to
    solve the puzzle
    """
    Image.open(puzzle_in_image).show()

    puzzle_dimensions = get_puzzle_dimensions(puzzle_in_image)
    print("puzzle dimensions:", puzzle_dimensions)

    puzzle_pieces = get_puzzle_pieces(puzzle_in_image, puzzle_dimensions)
    print("puzzle piece count:", len(puzzle_pieces))

    matches = get_puzzle_piece_matches(puzzle_pieces)
    print("puzzle matches count:", len(matches))

    solved_puzzle = move_puzzle_pieces(puzzle_dimensions, puzzle_pieces, matches)
    print("puzzle solved")

    generate_output_image(puzzle_dimensions, solved_puzzle)
    print("puzzle output generated")


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        print("RUNNING...")
        solve_puzzle(args[0])
    else:
        print("python jigsaw.py <puzzle_image_filename>")


if __name__ == '__main__':
    main()

