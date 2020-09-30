import numpy as np
import os


class BeamSectionDesigner(object):
    '''
    This object designs beam cross-sections
    '''

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

    def design(self):

        preferred_bar = self.preferred_bar
        m_demand = self.m_demand

        design_complete = 0
        considered_covers = [0.05, 0.06, 0.04, 0.07]

        # Section: PROVIDED INFO
        if m_demand[0] < 0.38 * m_demand[1]:  # As'>=0.38As CL 9.4.3.4
            m_demand[0] = 0.38 * m_demand[1]
        if m_demand[1] < 0.38 * m_demand[0]:
            m_demand[1] = 0.38 * m_demand[0]

        # Varied info:
        phi = 0.85
        J = 0.85

        alpha = max(0.75, 0.85 - 0.004 * max((self.fc / 1000000 - 55), 0))  # CL 7.4.2.7
        beta = max(0.65, 0.85 - 0.008 * max(self.fc / 1e6 - 30, 0))

        dbs = np.array([0.010, 0.012, 0.016, 0.020, 0.025, 0.032])
        areas = dbs ** 2 * np.pi / 4
        bar_forces = areas * self.fy
        n_sizes = len(dbs)

        bar_arrangement = [[], []]
        for rot in range(2):
            As_approx = self.m_demand[rot] / (phi * self.fy * (self.depth * J))

            if self.verbose == 1:
                print('As_approx: ', As_approx)
            c = As_approx * self.fy / (alpha * beta * self.fc * self.width)
            a = beta * c / 2
            d = self.depth - (0.06 + 0.065 + 0.06) / 2
            lever = d - a
            bar_options = []
            Force_req = self.m_demand[rot] / (phi * lever)

            c = Force_req / (alpha * beta * self.fc * self.width)
            a = beta * c / 2
            d = self.depth - (0.06 + 0.065 + 0.06) / 2
            lever = d - a
            Force_req = self.m_demand[rot] / (phi * lever)
            if self.verbose == 1:
                print('Force required', Force_req)
            for choice in range(len(considered_covers)):
                cover = considered_covers[choice]

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
                            bar_options.append({dbs[i]: int(n_bars)})
                            if j:
                                bar_options[-1][dbs[i - 1]] = j

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
                            bar_options.append({dbs[i]: int(n_bars)})
                            if j:
                                bar_options[-1][dbs[i + 1]] = j

                # Section: DESIGN CHECKS
                if self.verbose == 1:
                    print('\n \n bar_options: ', bar_options)
                Layer = [[], []]

                # for i in range(n_sizes - 2):
                good_bar_options = []
                for j in range(len(bar_options)):
                    nbd = bar_options[j]
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
                        print('Failed:', bar_options[j], ' Bar diameter too big')
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
                            print('Failed:', bar_options[j], ' Not enough corner bars')
                        continue

                    # print check
                    if sum(check) < 5:
                        continue
                    else:
                        good_bar_options.append(bar_options[j])
                bar_options = good_bar_options
                # Define bar spacing
                for j in range(len(bar_options)):
                    n_bars = np.sum([bar_options[j][db] for db in bar_options[j]])
                    av_db = np.sum([bar_options[j][db] * db for db in bar_options[j]]) / n_bars
                    max_db = max(list(bar_options[j]))
                    min_db = min(list(bar_options[j]))
                    loc0 = cover + np.ceil(max_db / 2 * 1e3) / 1e3
                    loc1 = loc0 + self.layer_spacing
                    nbd = bar_options[j]
                    if min_db == max_db:
                        min_db = None
                    bar_spacing = 0.06
                    bars_p_layer = ((self.width - 2 * cover) / (av_db + bar_spacing))
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
                        bar_arrangement[rot].append([Layer[0], Layer[1], loc0, loc1])
                        if self.verbose == 1:
                            print('Bar arrangement: ', [Layer[0], Layer[1], loc0, loc1])
                    else:  # 2-layers of bars
                        if self.verbose == 1:
                            print('need two layers of bars')
                        if min_db is None:  # only one bar type
                            if np.mod(nbd[max_db], 2) == 0:  # even same number of bars for both layers
                                Layer[0] = max_db * np.ones(int(nbd[max_db] / 2))
                                Layer[1] = max_db * np.ones(int(nbd[max_db] / 2))
                                bar_arrangement[rot].append([Layer[0], Layer[1], loc0, loc1])
                            else:  # add two combinations, one with extra bar on top and other with extra below
                                Layer[0] = max_db * np.ones(int(nbd[max_db] / 2 + 1))
                                Layer[1] = max_db * np.ones(int(nbd[max_db] / 2))
                                bar_arrangement[rot].append([Layer[0], Layer[1], loc0, loc1])
                                Layer[0] = max_db * np.ones(int(nbd[max_db] / 2))
                                Layer[1] = max_db * np.ones(int(nbd[max_db] / 2 + 1))
                                bar_arrangement[rot].append([Layer[0], Layer[1], loc0, loc1])
                        elif np.mod(nbd[max_db], 4) == 0:
                            if self.verbose == 1:
                                print('even number of main bars for top and bottom')
                            # arrange to have main bars on the outside and the additional bars in centre.
                            Layer[0] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                            Layer[1] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                            # Add smaller bars into the centre alternating each layer start with outer
                            for n in range(nbd[min_db]):
                                Layer[n % 2].insert(int(nbd[max_db] / 4), min_db)
                            bar_arrangement[rot].append([np.array(Layer[0]), np.array(Layer[1]), loc0, loc1])
                            Layer[0] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                            Layer[1] = list(max_db * np.ones(int(nbd[max_db] / 2)))
                            # Add smaller bars into the centre alternating each layer start with outer
                            for n in range(nbd[min_db]):
                                Layer[(n + 1) % 2].insert(int(nbd[max_db] / 4), min_db)
                            bar_arrangement[rot].append([np.array(Layer[0]), np.array(Layer[1]), loc0, loc1])

                        elif np.mod(nbd[max_db], 2) == 0:
                            # Add larger bars and add the extra two to the outer layer
                            Layer[0] = [max_db] * (int(nbd[max_db] / 4) * 2 + 2)
                            Layer[1] = [max_db] * int(nbd[max_db] / 4) * 2
                            for i in range(nbd[min_db]):
                                if i < 2:
                                    Layer[1].insert(int(len(Layer[1]) / 2), min_db)
                                else:
                                    Layer[i % 2].insert(int(len(Layer[i % 2]) / 2), min_db)
                            bar_arrangement[rot].append([np.array(Layer[0]), np.array(Layer[1]), loc0, loc1])
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
                                        n_left = int((len(Layer[k]) - 1) / 2)
                                        n_right = int((len(Layer[k]) + 1) / 2)
                                        Layer[k].insert(n_left, min_db)
                                        Layer[k].insert(n_right, min_db)
                                    else:
                                        # add to centre
                                        Layer[k].insert(int(len(Layer[k]) / 2), min_db)
                                        Layer[k].insert(int(len(Layer[k]) / 2), min_db)
                            # add remaining small bars to balance section
                            n_extra = nbd[min_db] % 4
                            for k in range(2):
                                if has_centre[k]:
                                    if n_extra == 1:
                                        continue
                                    elif n_extra >= 2:
                                        # add to left and right of main bar
                                        n_left = int((len(Layer[k]) - 1) / 2)
                                        n_right = int((len(Layer[k]) + 1) / 2)
                                        Layer[k].insert(n_left, min_db)
                                        Layer[k].insert(n_right, min_db)
                                else:
                                    if n_extra in [1, 3]:
                                        Layer[k].insert(int(len(Layer[k]) / 2), min_db)

                            bar_arrangement[rot].append([np.array(Layer[0]), np.array(Layer[1]), loc0, loc1])
                            if self.verbose == 1:
                                print('Bar arrangement: ', [Layer[0], Layer[1], loc0, loc1])

        # counting number of preferred_bar
        for attempt in range(2):
            score = [[], []]
            for rot in range(2):
                print('For rot ', rot, ' options available: %i' % len(bar_arrangement[rot]))
                if len(bar_arrangement[rot]) == 0:
                    print('M_stress ratio: ', self.m_demand[rot] / self.width / self.depth ** 2 / self.fc)

                for attempt in range(len(bar_arrangement[rot])):
                    count = 0
                    for i in range(len(bar_arrangement[rot][attempt][0])):
                        if bar_arrangement[rot][attempt][0][i] == preferred_bar:
                            count += 1
                    for i in range(len(bar_arrangement[rot][attempt][1])):
                        if bar_arrangement[rot][attempt][1][i] == preferred_bar:
                            count += 1
                    score[rot].append(count)

            if self.verbose == 1:
                print('SCORE: ', score)
            for vals in score:
                if max(vals) == 0:
                    print('no design with preferred bar size')
                    # get next preferred size
                    db_list = [0.010, 0.012, 0.016, 0.020, 0.025, 0.032]
                    pb = np.round(preferred_bar, 3)
                    db_ind = 100000
                    for dd in range(len(db_list)):
                        print(dd, pb)
                        if str(db_list[dd]) == str(pb):
                            print('found ya')
                            db_ind = dd
                    print('db_list: ', db_list)

                    # db_ind = db_list.index(pb)
                    preferred_bar = dbs[db_ind - 1]
                else:
                    break

        # print bar_options
        if self.verbose == 1:
            print('bar_arrangements: \n', bar_arrangement)

        BEAMPROPS = []
        LX = []
        MCAP = []
        for rot in range(2):  # len(self.m_demand)):
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
                c_block = force_T / (self.width * alpha * beta * self.fc)
                # print('c block: ', c_block

                Moment_cap = 0
                for layer in range(2):
                    Moment_cap = Moment_cap + phi * As_fy[layer] * (
                                self.depth - Beam_props[layer + 2] - c_block * beta / 2)
                if self.verbose == 1:
                    print('selected: ', select)
                    print('Moment Capacity: ', Moment_cap, 'Demand: ', self.m_demand[rot])
                if Moment_cap < 1.05 * self.m_demand[rot] and Moment_cap > 0.97 * self.m_demand[rot]:
                    check[0] = 1
                else:
                    # if self.verbose==1:
                    print('Failed moment check: ', ' Moment Capacity: %.1fkNm' % (Moment_cap / 1e3),
                          'Demand: %.1fkNm   %.1f%%' % (self.m_demand[rot] / 1e3, Moment_cap / self.m_demand[rot] * 100))

                # ##spacing check
                # spacing must be equal to or greater than max(db) or 25mm CL 8.31)
                min_width0 = sum(Layer[0]) + (len(Layer[0]) - 1) * max(Layer[0]) + 2 * cover
                min_width1 = sum(Layer[1]) + (len(Layer[1]) - 1) * max(Layer[1]) + 2 * cover
                if self.width > max(min_width0, min_width1):
                    check[1] = 1
                else:
                    if self.verbose == 1:
                        print('Failed on check CL 8.31')

                spacing = (self.width - 2 * cover) / (len(Beam_props[0]) - 1)
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
                          'Demand: %.1fkNm   %.1f%%' % (self.m_demand[rot] / 1000, Moment_cap / self.m_demand[rot] * 100),
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
                    Beam_props[layer + 2] = self.depth - Beam_props[layer + 2]
                for i in range(len(Beam_props[layer])):

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
                        sectfig.text(self.width + 0.10 * value + 0.05, Beam_props[layer + 2] - 0.015, diameter)
                    value += 1
            # Write moment capacity
            M_cap_str = 'Mn= \n' + str(float(int(Moment_cap / 100)) / 10) + 'KNm'
            sectfig.text(self.width + 0.05, self.depth / 2 - self.depth / 6 + rot * self.depth / 3,
                         M_cap_str)

            # Draw selected beam option:

            # draw beam edge:
            x_edge = [0, self.width, self.width, 0, 0]
            y_edge = [0, 0, self.depth, self.depth, 0]
            edge = sectfig.plot(x_edge, y_edge)
            plt.setp(edge, c='k', linewidth=1.5)

            # Centring and scaling image
            plot_size = max(self.width, self.depth) + 0.1
            # print plot_size
            extra = plot_size - self.width
            sectfig.axis('equal')
            sectfig.axis([-0.04 - extra / 2, self.width + extra / 2 + 0.04, -0.01, plot_size + 0.01])

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
    preferred_bar_diam = 0.025
    preferred_cover = 0.04
    layer_spacing = 0.04
    beam = BeamSectionDesigner(moment, depth, width, fc, fy, min_column_depth, preferred_bar_diam, preferred_cover, layer_spacing)
    beam.design()
    beam.plotSection()

