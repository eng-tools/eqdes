import numpy as np
import os


class BeamSection(object):
    '''
    This object designs beam cross-sections
    '''

    def __init__(self, M_star, Beam_depth, Beam_width, fc, fy, Min_Column_depth, given_prefered_bar, prefered_cover,
                 layer_spacing, **kwargs):
        '''
        Constructor
        '''
        self.M_star = M_star
        self.Beam_depth = Beam_depth
        self.Beam_width = Beam_width
        self.fc = fc
        self.fy = fy
        self.Min_Column_depth = Min_Column_depth
        self.given_prefered_bar = given_prefered_bar
        self.prefered_cover = prefered_cover
        self.layer_spacing = layer_spacing
        self.verbose = kwargs.get('verbose', 0)
        self.SectionName = kwargs.get('section_name', '')
        self.SavePath = kwargs.get('save_path', '')

    def design(self):

        prefered_bar = self.given_prefered_bar
        M_star = self.M_star

        design_complete = 0
        considered_covers = [0.05, 0.06, 0.04, 0.07]

        # Section: PROVIDED INFO
        if M_star[0] < 0.38 * M_star[1]:  # As'>=0.38As CL 9.4.3.4
            M_star[0] = 0.38 * M_star[1]
        if M_star[1] < 0.38 * M_star[0]:
            M_star[1] = 0.38 * M_star[0]

        # Varied info:
        phi = 0.85
        J = 0.85

        alpha = max(0.75, 0.85 - 0.004 * max((self.fc / 1000000 - 55), 0))  # CL 7.4.2.7
        beta = max(0.65, 0.85 - 0.008 * max(self.fc / 1e6 - 30, 0))

        db = np.array([0.010, 0.012, 0.016, 0.020, 0.025, 0.032])
        As_bar = db ** 2 * np.pi / 4
        Force_bar = As_bar * self.fy
        n_sizes = len(db)

        bar_arrangement = [[], []]
        for rot in range(2):
            As_approx = self.M_star[rot] / (phi * self.fy * (self.Beam_depth * J))

            if self.verbose == 1:
                print('As_approx: ', As_approx)
            c = As_approx * self.fy / (alpha * beta * self.fc * self.Beam_width)
            a = beta * c / 2
            d = self.Beam_depth - (0.06 + 0.065 + 0.06) / 2
            lever = d - a

            bar_options = []
            counter = 0 - 1
            Force_req = self.M_star[rot] / (phi * lever)

            c = Force_req / (alpha * beta * self.fc * self.Beam_width)
            a = beta * c / 2
            d = self.Beam_depth - (0.06 + 0.065 + 0.06) / 2
            lever = d - a
            Force_req = self.M_star[rot] / (phi * lever)
            if self.verbose == 1:
                print('Force required', Force_req)
            c = Force_req / (alpha * beta * self.fc * self.Beam_width)

            for choice in range(len(considered_covers)):
                cover = considered_covers[choice]

                for i in range(1, n_sizes - 1):
                    counter += 1
                    Location1 = cover + np.ceil(db[i + 1] / 2 * 1e3) / 1e3
                    Location2 = Location1 + self.layer_spacing
                    bar_options.append([])

                    # iterating over adding one smaller bar
                    for j in range(8):
                        Force_needed = Force_req - j * Force_bar[i - 1]
                        if Force_needed < 0:
                            break
                        left_over = Force_needed % Force_bar[i]
                        higher_r = Force_bar[i] - left_over
                        if left_over < higher_r:
                            n_bars = np.floor(Force_needed / Force_bar[i])
                            remainder = left_over
                        else:
                            n_bars = np.ceil(Force_needed / Force_bar[i])
                            remainder = higher_r
                        if remainder < 0.10 * Force_req:
                            bar_options[counter].append([j, int(n_bars), 0])

                    # iterating over adding larger bars:
                    for j in range(8):
                        Force_needed = Force_req - j * Force_bar[i + 1]
                        if Force_needed < 0:
                            break
                        left_over = Force_needed % Force_bar[i]
                        higher_r = Force_bar[i] - left_over
                        if left_over < higher_r:
                            n_bars = np.floor(Force_needed / Force_bar[i])
                            remainder = left_over
                        else:
                            n_bars = np.ceil(Force_needed / Force_bar[i])
                            remainder = higher_r
                        if remainder < 0.10 * Force_req:
                            bar_options[counter].append([0, int(n_bars), j])

                # Section: DESIGN CHECKS
                if self.verbose == 1:
                    print('\n \n bar_options: ', bar_options)
                Layer = [[], []]

                for i in range(n_sizes - 2):
                    for j in range(len(bar_options[i])):
                        if bar_options[i][j][0] > 0:
                            extra_b = 0
                        else:
                            extra_b = 2

                        check = np.zeros((5))

                        # #steel ratio
                        As_tot = bar_options[i][j][0] * db[i] ** 2 * np.pi / 4 + bar_options[i][j][1] * db[
                            i + 1] ** 2 * np.pi / 4 + bar_options[i][j][2] * db[i + 2] ** 2 * np.pi / 4
                        p_steel = As_tot / (self.Beam_width * self.Beam_depth)
                        # #min steel ratio CL 9.4.3.4
                        p_min = np.sqrt(self.fc) / (4 * self.fy)
                        if p_steel > p_min:
                            check[0] = 1
                        else:
                            if self.verbose == 1:
                                print('Failed:', db[i + 1], bar_options[i][j], ' Below minimum steel ratio')

                        # #max steel ratio
                        # gravity: the distance from the extreme compression fibre
                        # to the neutral axis is less than 0.75cb (CL 9.3.8.1)

                        p_max = min((self.fc / 1e6 + 10) / (6 * self.fy / 1e6), 0.025)  # CL 9.4.3.3
                        if p_steel < p_max:
                            check[1] = 1
                        elif self.verbose == 1:
                            print('Failed:', db[i + 1], bar_options[i][j], ' Exceeded maximum steel ratio')

                        # #max bar check
                        alpha_f = 1.0  # 1.0 for oneway frame, 0.85 for 2way frame
                        alpha_d = 1.0  # 1.0 for ductile connections and 1.2 in limited ductile
                        db_max = 3.3 * alpha_f * alpha_d * np.sqrt(self.fc / 1e6) / (
                                    1.25 * self.fy / 1e6) * self.Min_Column_depth
                        # print db_max

                        if bar_options[i][j][2] == 0:
                            largest_db = db[i + 1]
                        else:
                            largest_db = db[i + 2]

                        if db_max > largest_db:
                            check[2] = 1
                        elif self.verbose == 1:
                            check[2] = 1
                            print('Failed:', db[i + 1], bar_options[i][j], ' Bar diameter too big')
                            print('test disabled')

                        # #hook length
                        required_in_length = max(8 * largest_db, 0.2)

                        check[3] = 1

                        # Minimum steel
                        # need to have atleast 2 16mm bars top and bottom
                        if i == 0 and bar_options[i][j][2] < 2:
                            if self.verbose == 1:
                                print('Failed:', db[i + 1], bar_options[i][j], ' Not enough corner bars')
                        elif i == 1 and (bar_options[i][j][1] + bar_options[i][j][2]) < 2:
                            if self.verbose == 1:
                                print('Failed:', db[i + 1], bar_options[i][j], ' Not enough corner bars')
                        else:
                            check[4] = 1

                        # print check
                        if sum(check) < 5:
                            if self.verbose == 1:
                                print('failed checks ', db[i + 1], bar_options[i][j])
                            break
                        else:
                            # spacing
                            bars_p_layer = ((self.Beam_width - 2 * cover) / (db[i] + 0.060))
                            n_bars = sum(bar_options[i][j])
                            number_layers = n_bars / bars_p_layer
                            # print('Number of layers required: ',number_layers)
                            if n_bars > 2 and number_layers < 2.4:
                                # check[2]=1 #number of bars is suitable
                                if number_layers < 1.3:
                                    if self.verbose == 1:
                                        print('One layer of bars')

                                    if np.mod(bar_options[i][j][1], 2) == 0:
                                        if self.verbose == 1:
                                            print('even number of main bars')
                                        # arrange to have main bars on the outside and the additional bars in centre.
                                        Layer[0] = db[i + 1] * np.ones(int(bar_options[i][j][1]))
                                        Layer[0] = list(Layer[0])
                                        for k in range(bar_options[i][j][extra_b]):
                                            Layer[0].insert(int(bar_options[i][j][1] / 2), db[i + extra_b])
                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = 0 * (Layer[0])

                                        check_bars = sum(bar_options[i][j]) - len(Layer[0])
                                        if check_bars != 0:
                                            print('ERROR with' + str(bar_options[i][j]))
                                            print('check bars: ', check_bars)
                                            print('Not_working')
                                            raise ValueError()

                                        bar_arrangement[rot].append([Layer[0], Layer[1], Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                    elif np.mod(bar_options[i][j][1], 2) == 1 and np.mod(bar_options[i][j][extra_b],
                                                                                         2) == 1:
                                        if self.verbose == 1:
                                            print('Uneven bar arrangement, arrangement removed')

                                    else:
                                        if self.verbose == 1:
                                            print('uneven main bar numbers, even additional bars')
                                        Layer[0] = db[i + 1] * np.ones(int(bar_options[i][j][1]))
                                        Layer[0] = list(Layer[0])

                                        for side in range(2):
                                            for k in range(int(bar_options[i][j][extra_b] / 2)):
                                                Layer[0].insert(int(bar_options[i][j][1] / 2 + 1 - side), db[i + extra_b])

                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = 0 * (Layer[0])
                                        check_bars = sum(bar_options[i][j]) - (len(Layer[0]))
                                        if check_bars != 0:
                                            print('ERROR with' + str(bar_options[i][j]))
                                            print('check bars: ', check_bars)
                                            raise ValueError()

                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = np.array(Layer[1])
                                        bar_arrangement[rot].append([Layer[0], Layer[1], Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                else:
                                    if self.verbose == 1:
                                        print('need two layers of bars')
                                    if np.mod(bar_options[i][j][1], 4) == 0:
                                        if self.verbose == 1:
                                            print('even number of main bars for top and bottom')
                                        # arrange to have main bars on the outside and the additional bars in centre.
                                        for layer in range(2):
                                            Layer[layer] = db[i + 1] * np.ones(int(bar_options[i][j][1] / 2))
                                            Layer[layer] = list(Layer[layer])
                                            for k in range(int(bar_options[i][j][0] / 2)):
                                                Layer[layer].insert(int(bar_options[i][j][1] / 4), db[i])
                                            for k in range(int(bar_options[i][j][2] / 2)):
                                                Layer[layer].insert(int(bar_options[i][j][1] / 4), db[i + 2])

                                        if np.mod(bar_options[i][j][0], 2) == 1:
                                            # smaller bars added
                                            Layer[0].insert(int(bar_options[i][j][1] / 4), db[i])

                                        elif np.mod(bar_options[i][j][2], 2) == 1:
                                            Layer[0].insert(int(bar_options[i][j][1] / 4), db[i + 2])

                                        check_bars = sum(bar_options[i][j]) - (len(Layer[0]) + len(Layer[1]))
                                        if check_bars != 0:
                                            print('ERROR with' + str(bar_options[i][j]))
                                            raise ValueError()

                                        bar_arrangement[rot].append(
                                            [np.array(Layer[0]), np.array(Layer[1]), Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                    elif np.mod(bar_options[i][j][1], 2) == 0 and np.mod((bar_options[i][j][extra_b]),
                                                                                         4) == 0:
                                        if self.verbose == 1:
                                            print('split main bars top and bottom, and extras bars')
                                        for layer in range(2):

                                            Layer[layer] = db[i + 1] * np.ones(int(bar_options[i][j][1] / 2))
                                            Layer[layer] = list(Layer[layer])

                                            if self.verbose == 1:
                                                print('Layer info ', Layer)
                                            for side in range(2):
                                                for k in range(int(bar_options[i][j][extra_b] / 4)):
                                                    Layer[layer].insert(int(bar_options[i][j][1] / 4 + 1 - side), db[i + extra_b])

                                        check_bars = sum(bar_options[i][j]) - (len(Layer[0]) + len(Layer[1]))
                                        if check_bars != 0:
                                            print('ERROR with' + str(bar_options[i][j]))
                                            raise ValueError()

                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = np.array(Layer[1])
                                        bar_arrangement[rot].append([Layer[0], Layer[1], Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                    elif np.mod(bar_options[i][j][1], 2) == 0 and np.mod((bar_options[i][j][extra_b]),
                                                                                         2) == 0:
                                        if self.verbose == 1:
                                            print('split all bars top and bottom, with the larger number of bars going in the top layer')
                                        if bar_options[i][j][1] > bar_options[i][j][extra_b]:
                                            Layer[0] = db[i + 1] * np.ones(int(sum(bar_options[i][j]) / 2))
                                            Layer[1] = db[i + 1] * np.ones(int(sum(bar_options[i][j]) / 2 - bar_options[i][j][extra_b]))
                                            Layer[0] = list(Layer[0])
                                            Layer[1] = list(Layer[1])
                                            for k in range(bar_options[i][j][extra_b]):
                                                Layer[1].insert(int((sum(bar_options[i][j]) / 2 - bar_options[i][j][extra_b]) / 2), db[i + extra_b])
                                        else:
                                            Layer[0] = db[i + extra_b] * np.ones(int(sum(bar_options[i][j]) / 2))
                                            Layer[1] = db[i + extra_b] * np.ones(int((sum(bar_options[i][j]) / 2 - bar_options[i][j][1])))
                                            Layer[0] = list(Layer[0])
                                            Layer[1] = list(Layer[1])
                                            for k in range(bar_options[i][j][1]):
                                                Layer[1].insert(int((sum(bar_options[i][j]) / 2 - bar_options[i][j][1]) / 2), db[1])

                                        check_bars = sum(bar_options[i][j]) - (len(Layer[0]) + len(Layer[1]))
                                        if check_bars != 0:
                                            print('ERROR with' + str(bar_options[i][j]))
                                            raise ValueError()

                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = np.array(Layer[1])
                                        bar_arrangement[rot].append([Layer[0], Layer[1], Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                    elif np.mod(bar_options[i][j][1], 2) == 0:
                                        if self.verbose == 1:
                                            print(
                                                'split main bars with extra two in bottom layer, then add additional bars with extra bars going in top layer')
                                        Layer[0] = db[i + 1] * np.ones(int(bar_options[i][j][1] / 2 + 1))
                                        Layer[1] = db[i + 1] * np.ones(int(bar_options[i][j][1] / 2 - 1))

                                        for layer in range(2):
                                            Layer[layer] = list(Layer[layer])
                                            for k in range(int(bar_options[i][j][extra_b] / 2)):
                                                Layer[layer].insert(int((bar_options[i][j][1] / 2 + 1 - layer * -2) / 2), db[i + extra_b])

                                        # Add the extra bar into the top layer of bars
                                        if np.mod(bar_options[i][j][extra_b], 2) == 1:
                                            # smaller bars added
                                            Layer[1].insert(int((bar_options[i][j][1] / 2 - 1) / 2), db[i + extra_b])

                                        check_bars = sum(bar_options[i][j]) - (len(Layer[0]) + len(Layer[1]))
                                        if check_bars != 0:
                                            print('ERROR with' + str(bar_options[i][j]))
                                            print('check bars: ', check_bars)
                                            raise ValueError()

                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = np.array(Layer[1])
                                        bar_arrangement[rot].append([Layer[0], Layer[1], Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                    # ODD number of main bars and ODD number of main bars
                                    elif np.mod(bar_options[i][j][extra_b], 2) == 1:
                                        if self.verbose == 1:
                                            print('ODD number of main bars and ODD number of main bars')
                                            print(bar_options[i][j])
                                        diff = bar_options[i][j][1] - bar_options[i][j][extra_b]
                                        if np.mod(diff, 4) == 0:
                                            if self.verbose == 1:
                                                print('difference is a factor of 4')
                                            if diff > 0:
                                                Layer[0] = db[i + 1] * np.ones(int(bar_options[i][j][extra_b] + diff * 2 / 4))
                                                Layer[1] = db[i + extra_b] * np.ones(int(bar_options[i][j][extra_b] + diff * 2 / 4))
                                                for x in range(int(diff / 4)):
                                                    Layer[1][x] = db[i + 1]
                                                    Layer[1][-1 - x] = db[i + 1]
                                            else:
                                                Layer[0] = db[i + 1] * np.ones(int(bar_options[i][j][1] - diff / 2))
                                                Layer[1] = db[i + extra_b] * np.ones(int(bar_options[i][j][extra_b] - diff * 2 / 4))
                                                for x in range(int(diff / 4)):
                                                    Layer[0][x] = db[i + extra_b]
                                                    Layer[0][-1 - x] = db[i + extra_b]

                                        elif np.mod(diff, 2) == 0:
                                            if self.verbose == 1:
                                                print('difference is a factor of 2')
                                            sets = int(sum(bar_options[i][j]) / 4)
                                            extra = sum(bar_options[i][j]) - sets * 4
                                            if diff > 0:
                                                Layer[0] = db[i + 1] * np.ones(int(sets * 2 + 1))
                                                Layer[1] = db[i + 1] * np.ones(int(sets * 2 - 1 + extra))
                                                for x in range(int((bar_options[i][j][extra_b]) / 2)):
                                                    Layer[1][(sets * 2 - 1 + extra) / 2 - 1 - x] = db[i + extra_b]
                                                    Layer[1][(sets * 2 - 1 + extra) / 2 + 1 + x] = db[i + extra_b]

                                                check_bars = sum(bar_options[i][j]) - (len(Layer[0]) + len(Layer[1]))
                                                if check_bars != 0:
                                                    print('ERROR with' + str(bar_options[i][j]))
                                                    raise ValueError()
                                            else:
                                                Layer[0] = db[i + extra_b] * np.ones(int(sets * 2 + 1))
                                                Layer[1] = db[i + extra_b] * np.ones(int(sets * 2 - 1 + extra))
                                                for x in range(int(bar_options[i][j][1] / 2)):
                                                    Layer[1][(sets * 2 - 1 + extra) / 2 - x] = db[i + 1]
                                                    Layer[1][(sets * 2 - 1 + extra) / 2 + 1 + x] = db[i + 1]

                                                check_bars = sum(bar_options[i][j]) - (len(Layer[0]) + len(Layer[1]))
                                                if check_bars != 0:
                                                    print('ERROR with' + str(bar_options[i][j]))
                                                    print('Check bars: ', check_bars)
                                                    raise ValueError()

                                        Layer[0] = np.array(Layer[0])
                                        Layer[1] = np.array(Layer[1])
                                        bar_arrangement[rot].append([Layer[0], Layer[1], Location1, Location2])
                                        if self.verbose == 1:
                                            print('Bar arrangement: ', [Layer[0], Layer[1], Location1, Location2])
                                    else:
                                        if self.verbose == 1:
                                            print('No more bar arrangements available')

        # counting number of preferred_bar
        for attempt in range(2):
            score = [[], []]
            for rot in range(2):
                print('For rot ', rot, ' options available: %i' % len(bar_arrangement[rot]))
                if len(bar_arrangement[rot]) == 0:
                    print('M_stress ratio: ', self.M_star[rot] / self.Beam_width / self.Beam_depth ** 2 / self.fc)

                for attempt in range(len(bar_arrangement[rot])):
                    count = 0
                    for i in range(len(bar_arrangement[rot][attempt][0])):
                        if bar_arrangement[rot][attempt][0][i] == prefered_bar:
                            count += 1
                    for i in range(len(bar_arrangement[rot][attempt][1])):
                        if bar_arrangement[rot][attempt][1][i] == prefered_bar:
                            count += 1
                    score[rot].append(count)

            if self.verbose == 1:
                print('SCORE: ', score)
            for vals in score:
                if max(vals) == 0:
                    print('no design with prefered bar size')
                    # get next preferred size
                    db_list = [0.010, 0.012, 0.016, 0.020, 0.025, 0.032]
                    pb = np.round(prefered_bar, 3)
                    db_ind = 100000
                    for dd in range(len(db_list)):
                        print(dd, pb)
                        if str(db_list[dd]) == str(pb):
                            print('found ya')
                            db_ind = dd
                    print('db_list: ', db_list)

                    # db_ind = db_list.index(pb)
                    prefered_bar = db[db_ind - 1]
                else:
                    break

        # print bar_options
        if self.verbose == 1:
            print('bar_arrangements: \n', bar_arrangement)

        BEAMPROPS = []
        LX = []
        MCAP = []
        for rot in range(2):  # len(self.M_star)):
            if len(bar_arrangement[rot]) == 0:
                space_pass = 0
            for attempt in range(len(bar_arrangement[rot])):
                select = score[rot].index(max(score[rot]))

                Beam_props = bar_arrangement[rot][select]
                Layer = [Beam_props[0], Beam_props[1]]

                L_x = [[], []]

                # ADDITIONAL DESIGN CHECKS
                space_pass = 1
                check = np.zeros(2)
                # moment capacity check:
                As_fy = np.zeros(2)
                for layer in range(2):
                    As_fy[layer] = sum(Layer[layer] ** 2 * np.pi / 4 * self.fy)

                force_T = sum(As_fy)
                c_block = force_T / (self.Beam_width * alpha * beta * self.fc)
                # print('c block: ', c_block

                Moment_cap = 0
                for layer in range(2):
                    Moment_cap = Moment_cap + phi * As_fy[layer] * (
                                self.Beam_depth - Beam_props[layer + 2] - c_block * beta / 2)
                if self.verbose == 1:
                    print('selected: ', select)
                    print('Moment Capacity: ', Moment_cap, 'Demand: ', self.M_star[rot])
                if Moment_cap < 1.05 * self.M_star[rot] and Moment_cap > 0.97 * self.M_star[rot]:
                    check[0] = 1
                else:
                    # if self.verbose==1:
                    print('Failed moment check: ', ' Moment Capacity: %.1fkNm' % (Moment_cap / 1e3),
                          'Demand: %.1fkNm   %.1f%%' % (self.M_star[rot] / 1e3, Moment_cap / self.M_star[rot] * 100))

                # ##spacing check
                # spacing must be equal to or greater than max(db) or 25mm CL 8.31)
                min_width0 = sum(Layer[0]) + (len(Layer[0]) - 1) * max(Layer[0]) + 2 * cover
                min_width1 = sum(Layer[1]) + (len(Layer[1]) - 1) * max(Layer[1]) + 2 * cover
                if self.Beam_width > max(min_width0, min_width1):
                    check[1] = 1
                else:
                    if self.verbose == 1:
                        print('Failed on check CL 8.31')

                spacing = (self.Beam_width - 2 * cover) / (len(Beam_props[0]) - 1)
                if len(Beam_props[0]) == len(Beam_props[1]):
                    # even top and bottom, make same spacing
                    for layer in range(2):
                        for i in range(len(Beam_props[layer])):
                            L_x[layer].append(cover + i * spacing)
                elif len(Beam_props[0]) - len(Beam_props[1]) == 1:
                    if np.mod(len(Beam_props[0]), 2) == 1:
                        for layer in range(2):
                            for i in range(len(Beam_props[layer])):
                                extra = 0
                                if layer == 1 and i >= len(Beam_props[1]) / 2:
                                    extra = 1  # Double the middle spacing in the top layer
                                L_x[layer].append(cover + (i + extra) * spacing)

                    else:
                        if self.verbose == 1:
                            print('ERROR with spacing layout1: ', Beam_props)
                        space_pass = 0

                elif len(Beam_props[0]) - len(Beam_props[1]) == 2:
                    if np.mod(len(Beam_props[0]), 2) == 1:
                        for layer in range(2):
                            for i in range(len(Beam_props[layer])):
                                extra = 0
                                if layer == 1 and i == len(Beam_props[1]) / 2:
                                    extra = 1  # Double the spacing on the right in the top layer
                                elif layer == 1 and i >= len(Beam_props[1]) / 2 + 1:
                                    extra = 2
                                L_x[layer].append(cover + (i + extra) * spacing)
                    else:
                        for layer in range(2):
                            for i in range(len(Beam_props[layer])):
                                extra = 0
                                if layer == 1 and i >= len(Beam_props[1]) / 2:
                                    extra = 2  # Triple the middle spacing in the top layer

                                L_x[layer].append(cover + (i + extra) * spacing)

                else:
                    if self.verbose == 1:
                        print('ERROR with spacing layout: ', Beam_props)
                    space_pass = 0

                if space_pass == 1 and sum(check) == 2:
                    design_complete = 1
                    print('Layout rotation: ', str(rot), ' Moment Capacity: %.1fkNm' % (Moment_cap / 1000),
                          'Demand: %.1fkNm   %.1f%%' % (self.M_star[rot] / 1000, Moment_cap / self.M_star[rot] * 100),
                          ' ACCEPTED')
                    break
                else:
                    score[rot][select] = -1

            if space_pass == 0:
                print('Layout Error, no sufficient design options available')
                break
            if sum(check) != 2:
                print('checks: ', check)

            BEAMPROPS.append(Beam_props)
            LX.append(L_x)
            MCAP.append(Moment_cap)

        if self.verbose == 1:
            print('BEAMPROPS: \n', BEAMPROPS)
            print('LX: ', LX)
            print('MCAP: ', MCAP)

        self.Beam_data = [BEAMPROPS, LX, MCAP]  # Need to store both Beam_props
        return [self.Beam_data, design_complete, check]

    def plotSection(self, **kwargs):
        """
        This function plots the cross-section based on the design from design()
        """
        import matplotlib.pyplot as plt
        show_plot = kwargs.get('show_plot', 1)
        save_on = kwargs.get('save_on', 0)
        self.SavePath = kwargs.get('save_path', self.SavePath)
        self.SectionName = kwargs.get('section_name', self.SectionName)

        # Draw reinforcing
        BEAMPROPS = self.Beam_data[0]
        LX = self.Beam_data[1]
        MCAP = self.Beam_data[2]
        # print('Beam props layer:',Beam_props[layer]
        # print('L_x',L_x[layer]
        BIGFIG = plt.figure()
        sectfig = BIGFIG.add_subplot(111)
        rad = 361

        for rot in range(2):
            Beam_props = BEAMPROPS[rot]
            L_x = LX[rot]
            Moment_cap = MCAP[rot]
            for layer in range(2):  # CHANGE THIS
                bar_label = {}
                if rot == 1:
                    Beam_props[layer + 2] = self.Beam_depth - Beam_props[layer + 2]
                for i in range(len(Beam_props[layer])):
                    # draw reinforcing
                    x_bar = np.zeros((rad))
                    y_bar = np.zeros((rad))

                    circle1 = plt.Circle((L_x[layer][i], Beam_props[layer + 2]), Beam_props[layer][i] / 2, color='k')
                    sectfig.add_patch(circle1)
                    try:
                        bar_label[Beam_props[layer][i]] += 1
                    except:
                        KeyError()
                        bar_label[Beam_props[layer][i]] = 1

                value = 0
                for label in bar_label:

                    #            diameter=str(len(Beam_props[layer]))+'D'+str(int((Beam_props[layer][0]*1000)))
                    #            sectfig.text(Beam_width+0.05,Beam_props[layer+2]+0.03,diameter)
                    diameter = str(bar_label[label]) + '-D' + str(int(label * 1000))
                    if label != 0:
                        sectfig.text(self.Beam_width + 0.10 * value + 0.05, Beam_props[layer + 2] - 0.015, diameter)
                    value += 1
            # Write moment capacity
            M_cap_str = 'Mn= \n' + str(float(int(Moment_cap / 100)) / 10) + 'KNm'
            sectfig.text(self.Beam_width + 0.05, self.Beam_depth / 2 - self.Beam_depth / 6 + rot * self.Beam_depth / 3,
                         M_cap_str)

            # Draw selected beam option:

            # draw beam edge:
            x_edge = [0, self.Beam_width, self.Beam_width, 0, 0]
            y_edge = [0, 0, self.Beam_depth, self.Beam_depth, 0]
            edge = sectfig.plot(x_edge, y_edge)
            plt.setp(edge, c='k', linewidth=1.5)

            # Centring and scaling image
            plot_size = max(self.Beam_width, self.Beam_depth) + 0.1
            # print plot_size
            extra = plot_size - self.Beam_width
            sectfig.axis('equal')
            sectfig.axis([-0.04 - extra / 2, self.Beam_width + extra / 2 + 0.04, -0.01, plot_size + 0.01])

        sectfig.set_xlabel('Width (m)')
        sectfig.set_ylabel('Depth (m)')
        sectfig.set_title(self.SectionName)
        if save_on == 1:
            if not os.path.exists(self.SavePath):
                os.makedirs(self.SavePath)
            figure_name = self.SavePath + self.SectionName + '.png'
            BIGFIG.savefig(figure_name, format='png')
        if show_plot == 1:
            plt.show()

        del BIGFIG
        plt.clf()
        plt.close()


if __name__ == '__main__':
    moment = [250.0e3, 150.0e3]
    depth = 0.5
    width = 0.4
    fc = 30e6
    fy = 300e6
    min_column_depth = 0.5
    given_preferred_bar = 0.025
    preferred_cover = 0.04
    layer_spacing = 0.04
    beam = BeamSection(moment, depth, width, fc, fy, min_column_depth, given_preferred_bar, preferred_cover, layer_spacing)
    beam.design()
    beam.plotSection()

