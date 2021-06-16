import numpy as np
import os


class BeamSectionDesigner(object):
    """
    This object designs beam cross-sections
    """

    def __init__(self, m_demand, depth, width, f_c, f_y, min_col_depth, preferred_bar, preferred_cover,
                 layer_spacing, **kwargs):
        """
        Constructor

        Parameters
        ----------
        m_demand
        depth
        width
        f_c
        f_y
        min_col_depth
        preferred_bar
        preferred_cover
        layer_spacing
        kwargs
        """
        self.m_demand = m_demand
        self.depth = depth
        self.width = width
        self.fc = f_c
        self.fy = f_y
        self.min_col_depth = min_col_depth
        self.preferred_bar = preferred_bar
        self.preferred_cover = preferred_cover
        self.layer_spacing = layer_spacing
        self.verbose = kwargs.get('verbose', 0)
        self.SectionName = kwargs.get('section_name', '')
        self.SavePath = kwargs.get('save_path', '')
        self.bar_options = [[], []]  # phase 1
        self.bar_arrangements = [[], []]  # phase 2
        self.moment_capacities = [[], []]  # phase 3a
        self.steel_areas = [[], []]  # phase 3a
        self.bar_size_scores = [[], []]  # phase 3b
        self.x_layers = [[], []]  # [rot][layer][bar_i]  # phase 3c
        self.min_spacing = [[], []]  # phase 3c
        self.selected_x_layers = [[], []]
        self.selected_bar_arrangements = [[], []]
        self.selected_moment_capacities = [None, None]
        self.selected_layer_locs = [[], []]
        self.conc_cover = 0.06  # TODO: make input
        self.dbs = np.array([0.010, 0.012, 0.016, 0.020, 0.025, 0.032])
        self.alpha = max(0.75, 0.85 - 0.004 * max((self.fc / 1000000 - 55), 0))  # CL 7.4.2.7
        self.beta = max(0.65, 0.85 - 0.008 * max(self.fc / 1e6 - 30, 0))
        # Varied info:
        self.phi = 0.85
        self.j_len = 0.85
        self.phase0_balance_design_moments()
        for rot in range(2):
            self.phase1_collate_possible_bar_options(rot)
            self.phase1b_perform_prelim_design_checks(rot)
            self.phase2_define_bar_arrangements(rot)
            self.phase2b_check_min_bar_spacing(rot)
            self.phase3a_calc_moment_capacities_and_steel_areas(rot)
            self.phase3b_calc_bar_size_scores(rot)
            self.phase3c_define_bar_spacing(rot)
        self.phase4_select_preferred_bar_arrangement()

    def phase0_balance_design_moments(self):
        # Section: PROVIDED INFO
        if self.m_demand[0] < 0.38 * self.m_demand[1]:  # As'>=0.38As CL 9.4.3.4
            self.m_demand[0] = 0.38 * self.m_demand[1]
        if self.m_demand[1] < 0.38 * self.m_demand[0]:
            self.m_demand[1] = 0.38 * self.m_demand[0]

    def phase1_collate_possible_bar_options(self, rot):

        areas = self.dbs ** 2 * np.pi / 4
        bar_forces = areas * self.fy
        n_sizes = len(self.dbs)

        As_approx = self.m_demand[rot] / (self.phi * self.fy * (self.depth * self.j_len))

        if self.verbose == 1:
            print('As_approx: ', As_approx)
        c = As_approx * self.fy / (self.alpha * self.beta * self.fc * self.width)
        a = self.beta * c / 2
        d = self.depth - (0.06 + 0.065 + 0.06) / 2
        lever = d - a

        Force_req = self.m_demand[rot] / (self.phi * lever)

        c = Force_req / (self.alpha * self.beta * self.fc * self.width)
        a = self.beta * c / 2
        d = self.depth - (0.06 + 0.065 + 0.06) / 2
        lever = d - a
        Force_req = self.m_demand[rot] / (self.phi * lever)
        if self.verbose == 1:
            print('Force required', Force_req)

        for i in range(1, n_sizes - 1):
            # iterating over adding one smaller bar
            for j in range(8):
                Force_needed = Force_req - j * bar_forces[i - 1]
                if Force_needed < 0:
                    break
                left_over = Force_needed % bar_forces[i]
                higher_r = bar_forces[i] - left_over
                if left_over < higher_r:
                    n_bars = np.floor(Force_needed / bar_forces[i])
                    remainder = left_over
                else:
                    n_bars = np.ceil(Force_needed / bar_forces[i])
                    remainder = higher_r
                if remainder < 0.10 * Force_req:
                    self.bar_options[rot].append({self.dbs[i]: int(n_bars)})
                    if j:
                        self.bar_options[rot][-1][self.dbs[i - 1]] = j

            # iterating over adding larger bars:
            for j in range(8):
                Force_needed = Force_req - j * bar_forces[i + 1]
                if Force_needed < 0:
                    break
                left_over = Force_needed % bar_forces[i]
                higher_r = bar_forces[i] - left_over
                if left_over < higher_r:
                    n_bars = np.floor(Force_needed / bar_forces[i])
                    remainder = left_over
                else:
                    n_bars = np.ceil(Force_needed / bar_forces[i])
                    remainder = higher_r
                if remainder < 0.10 * Force_req:
                    self.bar_options[rot].append({self.dbs[i]: int(n_bars)})
                    if j:
                        self.bar_options[rot][-1][self.dbs[i + 1]] = j

        # Section: DESIGN CHECKS
        if self.verbose == 1:
            print('\n \n bar_options: ', self.bar_options[rot])

    def phase1b_perform_prelim_design_checks(self, rot):

        # bar_options = self.bar_options[rot]

        # for i in range(n_sizes - 2):
        good_bar_options = []
        for j in range(len(self.bar_options[rot])):
            nbd = self.bar_options[rot][j]
            max_db = max(list(nbd))

            check = np.zeros(5)  # TODO: make as a dict

            # steel ratio
            As_tot = np.sum([nbd[db] * db ** 2 * np.pi / 4 for db in nbd])
            p_steel = As_tot / (self.width * self.depth)
            # #min steel ratio CL 9.4.3.4
            p_min = np.sqrt(self.fc) / (4 * self.fy)
            if p_steel > p_min:
                check[0] = 1
            else:
                if self.verbose == 1:
                    print('Failed:', nbd, ' Below minimum steel ratio')
                continue

            # max steel ratio
            # gravity: the distance from the extreme compression fibre
            # to the neutral axis is less than 0.75cb (CL 9.3.8.1)

            p_max = min((self.fc / 1e6 + 10) / (6 * self.fy / 1e6), 0.025)  # CL 9.4.3.3
            if p_steel < p_max:
                check[1] = 1
            else:
                if self.verbose == 1:
                    print('Failed:', nbd, ' Exceeded maximum steel ratio')
                continue
            # max bar check
            alpha_f = 1.0  # 1.0 for oneway frame, 0.85 for 2way frame
            alpha_d = 1.0  # 1.0 for ductile connections and 1.2 in limited ductile
            db_max_lim = 3.3 * alpha_f * alpha_d * np.sqrt(self.fc / 1e6) / (1.25 * self.fy / 1e6) * self.min_col_depth
            # print db_max

            if db_max_lim > max_db:
                check[2] = 1
            elif self.verbose == 1:
                check[2] = 1  # TODO: test is disabled
                print('Failed:', nbd, ' Bar diameter too big')
                print('test disabled')

            # hook length
            required_in_length = max(8 * max_db, 0.2)
            check[3] = 1  # TODO:

            # Minimum steel
            # need to have at least 2 16mm bars top and bottom
            n_big_bars = np.sum([nbd[db] for db in nbd if db >= 0.016])
            if n_big_bars > 2:
                check[4] = 1
            else:
                if self.verbose == 1:
                    print('Failed:', nbd, ' Not enough corner bars')
                continue

            # print check
            if sum(check) < 5:
                continue
            else:
                good_bar_options.append(nbd)
        self.bar_options[rot] = good_bar_options

    def phase2_define_bar_arrangements(self, rot):
        bar_options = self.bar_options[rot]
        Layer = [[], []]
        bar_spacing = 0.06
        # Define bar spacing
        for j in range(len(bar_options)):
            n_bars = np.sum([bar_options[j][db] for db in bar_options[j]])
            av_db = np.sum([bar_options[j][db] * db for db in bar_options[j]]) / n_bars
            max_db = max(list(bar_options[j]))
            min_db = min(list(bar_options[j]))
            nbd = bar_options[j]
            if min_db == max_db:
                min_db = None
            bars_p_layer = ((self.width - 2 * bar_spacing) / (av_db + bar_spacing))
            number_layers = n_bars / bars_p_layer
            if n_bars < 2 or number_layers < 2.4:
                continue
            if number_layers < 1.3:
                if self.verbose == 1:
                    print('One layer of bars')
                if min_db is None:
                    Layer[0] = max_db * np.ones(nbd[max_db])
                elif np.mod(nbd[max_db], 2) == 0:
                    if self.verbose == 1:
                        print('even number of large bars')
                    # arrange to have main bars on the outside and the additional bars in centre.
                    Layer[0] = list(max_db * np.ones(nbd[max_db]))
                    for k in range(nbd[min_db]):
                        Layer[0].insert(int(nbd[max_db] / 2), min_db)
                    Layer[0] = np.array(Layer[0])
                    assert n_bars - len(Layer[0]) == 0
                elif np.mod(nbd[max_db], 2) == 1 and min_db is not None and np.mod(nbd[min_db], 2) == 1:
                    if self.verbose == 1:
                        print('Uneven bar arrangement, arrangement removed')
                        continue
                else:
                    if self.verbose == 1:
                        print('uneven large bar numbers, even additional bars')
                    Layer[0] = list(max_db * np.ones(nbd[max_db]))
                    for side in range(2):
                        for k in range(int(nbd[min_db] / 2)):
                            Layer[0].insert(int(nbd[max_db] / 2 + 1 - side), min_db)
                    Layer[0] = np.array(Layer[0])
                    assert n_bars - len(Layer[0]) == 0
                Layer[1] = np.ones_like(Layer[0])
                self.bar_arrangements[rot].append([Layer[0], Layer[1]])

            else:  # 2-layers of bars
                if self.verbose == 1:
                    print('need two layers of bars')
                if min_db is None:  # only one bar type
                    if np.mod(nbd[max_db], 2) == 0:  # even same number of bars for both layers
                        Layer[0] = max_db * np.ones(int(nbd[max_db] / 2))
                        Layer[1] = max_db * np.ones(int(nbd[max_db] / 2))
                        self.bar_arrangements[rot].append([Layer[0], Layer[1]])
                    else:  # add two combinations, one with extra bar on top and other with extra below
                        Layer[0] = max_db * np.ones(int(nbd[max_db] / 2 + 1))
                        Layer[1] = max_db * np.ones(int(nbd[max_db] / 2))
                        self.bar_arrangements[rot].append([Layer[0], Layer[1]])
                        Layer[0] = max_db * np.ones(int(nbd[max_db] / 2))
                        Layer[1] = max_db * np.ones(int(nbd[max_db] / 2 + 1))
                        self.bar_arrangements[rot].append([Layer[0], Layer[1]])
                elif np.mod(nbd[max_db], 4) == 0:
                    if self.verbose == 1:
                        print('even number of main bars for top and bottom')
                    # arrange to have main bars on the outside and the additional bars in centre.
                    Layer[0] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                    Layer[1] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                    # Add smaller bars into the centre alternating each layer start with outer
                    for n in range(nbd[min_db]):
                        Layer[n % 2].insert(int(nbd[max_db] / 4), min_db)
                    self.bar_arrangements[rot].append([np.array(Layer[0]), np.array(Layer[1])])
                    Layer[0] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                    Layer[1] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                    # Add smaller bars into the centre alternating each layer start with outer
                    for n in range(nbd[min_db]):
                        Layer[(n + 1) % 2].insert(int(nbd[max_db] / 4), min_db)
                    self.bar_arrangements[rot].append([np.array(Layer[0]), np.array(Layer[1])])

                elif np.mod(nbd[max_db], 2) == 0:
                    # Add larger bars and add the extra two to the outer layer
                    Layer[0] = [max_db] * (int(nbd[max_db] / 4) * 2 + 2)
                    Layer[1] = [max_db] * int(nbd[max_db] / 4) * 2
                    for i in range(nbd[min_db]):
                        if i < 2:
                            Layer[1].insert(int(len(Layer[1]) / 2), min_db)
                        else:
                            Layer[i % 2].insert(int(len(Layer[i % 2]) / 2), min_db)
                    self.bar_arrangements[rot].append([np.array(Layer[0]), np.array(Layer[1])])
                # elif np.mod(nbd[min_db], 2) != 0:
                #     continue
                elif nbd[max_db] > 0:  # uneven main bars greater than 4?
                    if nbd[max_db] % 4 == 1:
                        has_centre = [1, 0]
                        # Add larger bars and add the extra bar to the outer layer
                        Layer[0] = [max_db] * (int(nbd[max_db] / 4) * 2 + 1)
                        Layer[1] = [max_db] * int(nbd[max_db] / 4) * 2
                    else:  # has 3 extra
                        has_centre = [0, 1]
                        # Add larger bars and add the extra bar to the outer layer
                        Layer[0] = [max_db] * (int(nbd[max_db] / 4) * 2 + 2)
                        Layer[1] = [max_db] * (int(nbd[max_db] / 4) * 2 + 1)
                    # Then alternate adding smaller bars two at a time
                    for i in range(int(nbd[min_db] / 4)):
                        for k in range(2):
                            if has_centre[k]:
                                # add to left and right of main bar
                                n_left = int((len(Layer[k])) / 2)
                                n_right = int((len(Layer[k])) / 2 + 1)
                                Layer[k].insert(n_right, min_db)
                                Layer[k].insert(n_left, min_db)
                            else:
                                # add to centre
                                Layer[k].insert(int(len(Layer[k]) / 2), min_db)
                                Layer[k].insert(int(len(Layer[k]) / 2), min_db)
                    # add remaining small bars to balance section
                    n_extra = nbd[min_db] % 4
                    # note that layer0 len is always greater than layer1
                    if n_extra == 1:
                        if has_centre[0]:  # then add to layer[1]
                            Layer[1].insert(int(len(Layer[1]) / 2), min_db)
                        else:  # now layer[0] has two more bars than layer [1]
                            Layer[0].insert(int(len(Layer[0]) / 2), min_db)
                    elif n_extra == 2:  # now layer[1] has 1 extra - not a great design
                        if has_centre[1]:
                            n_left = int((len(Layer[1])) / 2)
                            n_right = int((len(Layer[1])) / 2 + 1)
                        else:
                            n_left = int((len(Layer[1])) / 2)
                            n_right = int((len(Layer[1])) / 2)
                        Layer[1].insert(n_right, min_db)
                        Layer[1].insert(n_left, min_db)
                    elif n_extra == 3:
                        if has_centre[0]:  # then add 1 to layer[1] and 2 to layer[0]
                            Layer[1].insert(int(len(Layer[1]) / 2), min_db)
                            n_left = int((len(Layer[0])) / 2)
                            n_right = int((len(Layer[0])) / 2 + 1)
                            Layer[0].insert(n_right, min_db)
                            Layer[0].insert(n_left, min_db)
                        else:
                            Layer[0].insert(int(len(Layer[0]) / 2), min_db)
                            n_left = int((len(Layer[1])) / 2)
                            n_right = int((len(Layer[1])) / 2 + 1)
                            Layer[1].insert(n_right, min_db)
                            Layer[1].insert(n_left, min_db)

                    if abs(len(Layer[0]) - len(Layer[1])) > 2:
                        raise ValueError(f'{rot}, {i}')
                    self.bar_arrangements[rot].append([np.array(Layer[0]), np.array(Layer[1])])

    def phase2b_check_min_bar_spacing(self, rot):
        # spacing must be equal to or greater than max(db) or 25mm CL 8.31)
        to_remove = []
        for i in range(len(self.bar_arrangements[rot])):
            layers = self.bar_arrangements[rot][i]
            min_width0 = sum(layers[0]) + (len(layers[0]) - 1) * max(layers[0]) + 2 * self.conc_cover
            min_width1 = sum(layers[1]) + (len(layers[1]) - 1) * max(layers[1]) + 2 * self.conc_cover
            if self.width < max(min_width0, min_width1):
                to_remove.append(i)
        for i in to_remove:
            del self.bar_arrangements[rot][i]

    def phase3b_calc_bar_size_scores(self, rot):
        # count number of preferred_bar
        # 1e5 for preferred_bar
        # 1e3 for lower size
        # 1 for bigger size
        pscore = 1e5
        bscore = 1
        sscore = 1e3
        ind = np.where(self.dbs == self.preferred_bar)[0][0]
        if ind + 1 == len(self.dbs):
            bigger_db = -1
        else:
            bigger_db = self.dbs[ind + 1]
        if ind - 1 == -1:
            smaller_db = -1
        else:
            smaller_db = self.dbs[ind - 1]
        score = []
        print('For rot ', rot, ' options available: %i' % len(self.bar_arrangements[rot]))
        if len(self.bar_arrangements[rot]) == 0:
            print('M_stress ratio: ', self.m_demand[rot] / self.width / self.depth ** 2 / self.fc)

        for opt in range(len(self.bar_arrangements[rot])):
            count = np.sum(np.where(self.bar_arrangements[rot][opt][0] == self.preferred_bar)) * pscore
            count += np.sum(np.where(self.bar_arrangements[rot][opt][1] == self.preferred_bar)) * pscore
            count += np.sum(np.where(self.bar_arrangements[rot][opt][0] == smaller_db)) * sscore
            count += np.sum(np.where(self.bar_arrangements[rot][opt][1] == smaller_db)) * sscore
            count += np.sum(np.where(self.bar_arrangements[rot][opt][0] == bigger_db)) * bscore
            count += np.sum(np.where(self.bar_arrangements[rot][opt][1] == bigger_db)) * bscore
            n_total = len(self.bar_arrangements[rot][opt][0]) + len(self.bar_arrangements[rot][opt][1])
            score.append(count / n_total)

        if self.verbose == 1:
            print('SCORE: ', score)
        if max(score) == 0:
            raise ValueError(f"No workable designs using the preferred bar diameter: {self.preferred_bar}, rot: {rot}")
        self.bar_size_scores[rot] = np.array(score)

    def phase3a_calc_moment_capacities_and_steel_areas(self, rot):

        for i in range(len(self.bar_arrangements[rot])):
            layers = self.bar_arrangements[rot][i]
            max_db_l0 = max(layers[0])
            locs = [self.conc_cover + np.ceil(max_db_l0 / 2 * 1e3) / 1e3]
            locs.append(locs[0] + self.layer_spacing)
            area_by_fy = np.zeros(2)
            steel_area = 0
            for a in range(2):
                steel_area += np.sum(layers[a] ** 2 * np.pi / 4)
                area_by_fy[a] = sum(layers[a] ** 2 * np.pi / 4 * self.fy)
            tension_force = sum(area_by_fy)
            c_block = tension_force / (self.width * self.alpha * self.beta * self.fc)

            moment_cap = 0
            for a in range(2):
                moment_cap += self.phi * area_by_fy[a] * (self.depth - locs[a] - c_block * self.beta / 2)
            self.steel_areas[rot].append(steel_area)
            self.moment_capacities[rot].append(moment_cap)
        self.steel_areas[rot] = np.array(self.steel_areas[rot])
        self.moment_capacities[rot] = np.array(self.moment_capacities[rot])

    def phase3c_define_bar_spacing(self, rot):
        for i in range(len(self.bar_arrangements[rot])):
            self.x_layers[rot].append([None, None])
            layers = self.bar_arrangements[rot][i]
            nbars0 = len(layers[0])
            nbars1 = len(layers[1])
            left_pos = self.conc_cover - max([layers[0][0], layers[0][0]]) / 2
            # Design layer with most bars first
            a = int(np.argmax([nbars0, nbars1]))
            x = np.linspace(left_pos, self.width - left_pos, len(layers[a]))
            # put at increments of 0.005 except centre always stay in centre
            x_half = x[:int(len(x) / 2)]
            x_half = np.round(x_half * 2, 2) / 2
            e = len(x) % 2
            x[:int(len(x) / 2)] = x_half
            x[int(len(x) / 2) + e:] = self.width - x_half[::-1]
            self.x_layers[rot][i][a] = x

            b = (a + 1) % 2
            if nbars0 == nbars1:  # if same number of bars use same layout
                self.x_layers[rot][i][b] = x
            elif len(layers[a]) - len(layers[b]) == 1:
                if len(layers[a]) % 2:  # larger layer has a centre bar, smaller layer should omit the centre bar
                    self.x_layers[rot][i][b] = np.delete(x, [int(len(x) / 2)])
                else:   # larger layer does not have a centre bar, smaller layer should have centre
                    inds = [int(len(x) / 2) - 1, int(len(x) / 2)]
                    x_centre = np.mean(x[inds])
                    xb = np.delete(x, [int(len(x) / 2)])
                    xb[int(len(xb) / 2)] = x_centre
                    self.x_layers[rot][i][b] = xb
            elif len(layers[a]) - len(layers[b]) == 2:
                if len(layers[a]) % 2:  # larger layer has a centre bar, smaller layer should keep centre bar
                    inds = [int(len(x) / 2) - 1, int(len(x) / 2) + 1]
                    self.x_layers[rot][i][b] = np.delete(x, inds)
                else:  # larger layer does not have a centre bar, smaller layer remove two central bars
                    inds = [int(len(x) / 2) - 1, int(len(x) / 2)]
                    self.x_layers[rot][i][b] = np.delete(x, inds)
            else:
                print(i, rot, layers[a], layers[b])
                raise ValueError()  # issue with initial bar arrangement!

            min_space = [100, 100]
            for a in range(2):
                x = self.x_layers[rot][i][a]
                min_space[a] = np.min(np.diff(x) - (layers[a][1:] + layers[a][:-1]) / 2)
            self.min_spacing[rot].append(np.min(min_space))
        self.min_spacing[rot] = np.array(self.min_spacing[rot])

    def phase4_select_preferred_bar_arrangement(self):
        rot = int(np.argmax(self.m_demand))  # select largest moment first
        bar_spacing_check = np.zeros(len(self.bar_arrangements[rot]))
        moment_capacity_check = np.zeros(len(self.bar_arrangements[rot]))
        for i in range(len(self.bar_arrangements[rot])):
            layers = self.bar_arrangements[rot][i]
            for a in range(2):
                max_db = max(layers[a])
                # spacing must be equal to or greater than max(db) or 25mm CL 8.31)
                if self.min_spacing[rot][i] > max([max_db, 0.025]):
                    bar_spacing_check[i] = 1

            if 0.98 * self.m_demand[rot] > self.moment_capacities[rot][i]:
                moment_capacity_check[i] = 0
            if 1.0 * self.m_demand[rot] > self.moment_capacities[rot][i]:
                moment_capacity_check[i] = 0.5
            elif self.moment_capacities[rot][i] < self.m_demand[rot] * 1.05:  # within 5% of target moment capacity
                moment_capacity_check[i] = 1
            elif self.moment_capacities[rot][i] < self.m_demand[rot] * 1.1:  # within 10% still worth picking for bar size
                moment_capacity_check[i] = 0.1
            elif self.moment_capacities[rot][i] < self.m_demand[rot] * 1.2:  # within 20% would rather pick diff bar size
                moment_capacity_check[i] = 0.001
            else:
                moment_capacity_check[i] = 0.00001
        overall_score = bar_spacing_check * moment_capacity_check * self.bar_size_scores[rot]
        selected_ind = int(np.argmax(overall_score))
        self.selected_x_layers[rot] = self.x_layers[rot][selected_ind]
        self.selected_moment_capacities[rot] = self.moment_capacities[rot][selected_ind]
        self.selected_bar_arrangements[rot] = self.bar_arrangements[rot][selected_ind]
        layers = self.selected_bar_arrangements[rot]
        max_db_l0 = max(layers[0])
        self.selected_layer_locs[rot] = [self.conc_cover + np.ceil(max_db_l0 / 2 * 1e3) / 1e3]
        self.selected_layer_locs[rot].append(self.selected_layer_locs[rot][0] + self.layer_spacing)
        selected_steel_area = self.steel_areas[rot][selected_ind]
        # Other direction
        rot2 = (rot + 1) % 2
        bar_spacing_check = np.zeros(len(self.bar_arrangements[rot2]))
        moment_capacity_check = np.zeros(len(self.bar_arrangements[rot2]))
        area_steel_check = np.zeros(len(self.bar_arrangements[rot2]))
        for i in range(len(self.bar_arrangements[rot2])):
            layers = self.bar_arrangements[rot2][i]
            for a in range(2):
                max_db = max(layers[a])
                # spacing must be equal to or greater than max(db) or 25mm CL 8.31)
                if self.min_spacing[rot2][i] > max([max_db, 0.025]):
                    bar_spacing_check[i] = 1

            if 0.98 * self.m_demand[rot2] > self.moment_capacities[rot2][i]:
                moment_capacity_check[i] = 0
            if 1.0 * self.m_demand[rot2] > self.moment_capacities[rot2][i]:
                moment_capacity_check[i] = 0.5
            elif self.moment_capacities[rot2][i] < self.m_demand[rot2] * 1.05:  # within 5% of target moment capacity
                moment_capacity_check[i] = 1
            elif self.moment_capacities[rot2][i] < self.m_demand[rot2] * 1.1:  # within 10% still worth picking for bar size
                moment_capacity_check[i] = 0.1
            elif self.moment_capacities[rot2][i] < self.m_demand[rot2] * 1.5:  # within 50% would rather pick diff bar size
                moment_capacity_check[i] = 0.001
            else:
                moment_capacity_check[i] = 0.00001
            if self.steel_areas[rot2][i] >= 0.38 * selected_steel_area:
                area_steel_check[i] = 1

        overall_score = bar_spacing_check * moment_capacity_check * self.bar_size_scores[rot2] * area_steel_check
        selected_ind = int(np.argmax(overall_score))
        self.selected_x_layers[rot2] = self.x_layers[rot2][selected_ind]
        self.selected_moment_capacities[rot2] = self.moment_capacities[rot2][selected_ind]
        self.selected_bar_arrangements[rot2] = self.bar_arrangements[rot2][selected_ind]
        layers = self.selected_bar_arrangements[rot]
        max_db_l0 = max(layers[0])
        self.selected_layer_locs[rot2] = [self.conc_cover + np.ceil(max_db_l0 / 2 * 1e3) / 1e3]
        self.selected_layer_locs[rot2].append(self.selected_layer_locs[rot2][0] + self.layer_spacing)

    def plot_section(self, **kwargs):
        """
        This function plots the cross-section based on the design from design()
        """
        import matplotlib.pyplot as plt
        show_plot = kwargs.get('show_plot', 1)
        save_on = kwargs.get('save_on', 0)
        self.SavePath = kwargs.get('save_path', self.SavePath)
        self.SectionName = kwargs.get('section_name', self.SectionName)

        # Draw reinforcing
        beam_props = self.selected_bar_arrangements
        layer_locs = self.selected_layer_locs
        lx = self.selected_x_layers
        m_cap = self.selected_moment_capacities
        # print('Beam props layer:',Beam_props[layer]
        # print('L_x',L_x[layer]
        bf = plt.figure()
        splot = bf.add_subplot(111)
        rad = 361

        for rot in range(2):
            beam_props_r = beam_props[rot]
            lxr = lx[rot]
            mom_cap_r = m_cap[rot]
            for layer in range(2):  # CHANGE THIS
                bar_label = {}
                if rot == 1:
                    y = self.depth - layer_locs[rot][layer]
                else:
                    y = layer_locs[rot][layer]
                for i in range(len(beam_props_r[layer])):

                    circle1 = plt.Circle((lxr[layer][i], y), beam_props_r[layer][i] / 2, color='k')
                    splot.add_patch(circle1)
                    bsize = beam_props_r[layer][i]
                    if bsize not in bar_label:
                        bar_label[bsize] = 0
                    bar_label[bsize] += 1

                value = 0
                for label in bar_label:
                    diameter = str(bar_label[label]) + '-D' + str(int(label * 1000))
                    if label != 0:
                        splot.text(self.width + 0.10 * value + 0.05, y - 0.015, diameter)
                    value += 1
            # Write moment capacity
            m_cap_str = 'Mn= \n' + str(float(int(mom_cap_r / 100)) / 10) + 'kNm'
            splot.text(self.width + 0.05, self.depth / 2 - self.depth / 6 + rot * self.depth / 3,
                         m_cap_str)

            # Draw selected beam option:

            # draw beam edge:
            x_edge = [0, self.width, self.width, 0, 0]
            y_edge = [0, 0, self.depth, self.depth, 0]
            edge = splot.plot(x_edge, y_edge)
            plt.setp(edge, c='k', linewidth=1.5)

            # Centring and scaling image
            plot_size = max(self.width, self.depth) + 0.1
            # print plot_size
            extra = plot_size - self.width
            splot.axis('equal')
            splot.axis([-0.04 - extra / 2, self.width + extra / 2 + 0.04, -0.01, plot_size + 0.01])

        splot.set_xlabel('Width [m]')
        splot.set_ylabel('Depth [m]')
        splot.set_title(self.SectionName)
        if save_on == 1:
            if not os.path.exists(self.SavePath):
                os.makedirs(self.SavePath)
            figure_name = self.SavePath + self.SectionName + '.png'
            bf.savefig(figure_name, format='png')
        if show_plot == 1:
            plt.show()

        del bf
        plt.clf()
        plt.close()


# if __name__ == '__main__':
#     moment = [250.0e3, 150.0e3]
#     depth = 0.5
#     width = 0.4
#     fc = 30e6
#     fy = 300e6
#     min_column_depth = 0.5
#     preferred_bar_diam = 0.02
#     preferred_cover = 0.04
#     layer_spacing = 0.04
#     beam = BeamSectionDesigner(moment, depth, width, fc, fy, min_column_depth, preferred_bar_diam, preferred_cover, layer_spacing)
#     beam.plot_section()

if __name__ == '__main__':
    moment = [230.0e3, 230.0e3]
    depth = 0.5
    width = 0.35
    fc = 30e6
    fy = 300e6
    min_column_depth = 0.5
    preferred_bar_diam = 0.025
    preferred_cover = 0.04
    layer_spacing = 0.04
    beam = BeamSectionDesigner(moment, depth, width, fc, fy, min_column_depth, preferred_bar_diam, preferred_cover, layer_spacing)
    beam.plot_section()