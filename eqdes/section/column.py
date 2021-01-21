import numpy as np


class ColumnSection(object):
    """
    This object designs beam cross-sections
    """
    def __init__(self, M_star, N_star, Column_depth, Column_width, fc, fy, E_s, given_prefered_bar, **kwargs):
        """
        Constructor

        Parameters
        ----------
        M_star
        N_star: currently only takes gravity load, need to account for seismic axial
        Column_depth
        Column_width
        fc
        fy
        E_s
        given_prefered_bar
        kwargs
        """
        self.M_star = M_star
        self.N_star = N_star
        self.Column_depth = Column_depth
        self.Column_width = Column_width
        self.fc = fc
        self.fy = fy
        self.E_s = E_s
        self.given_prefered_bar = given_prefered_bar
        #        self.prefered_cover=prefered_cover
        self.verbose = kwargs.get('verbose', 0)
        self.SectionName = kwargs.get('section_name', '')
        self.SavePath = kwargs.get('save_path', '')

    def design(self):
        """
        Design the cross-section according to NZS3101
        """

        verbose = 1

        if verbose == 1:
            print('M_star: ', self.M_star)
            print('N_star: ', self.N_star)
        #################
        # Varied info:
        phi = 1.0

        J = 0.8

        alpha = max(0.75, 0.85 - 0.004 * max((self.fc / 1000000 - 55), 0))  # CL 7.4.2.7
        beta = max(0.65, 0.85 - 0.008 * max(self.fc / 1e6 - 30, 0))

        db = np.array([0.010, 0.012, 0.016, 0.020, 0.025, 0.032])
        As_bar = db ** 2 * np.pi / 4
        Force_bar = As_bar * self.fy
        # print Force_bar
        n_sizes = len(db)

        # Section: BEGIN DESIGN
        design_complete = 0

        # Get coordinates for chart:
        M_coord = self.M_star / (self.Column_width * self.Column_depth ** 2)
        N_coord = self.N_star / (self.Column_width * self.Column_depth)

        N_nuet = [0.45, 0.3, 0.2]
        for trial in range(3):
            M_axial_approx = self.N_star * (N_nuet[trial] * self.Column_depth)
            As_approx = (self.M_star - M_axial_approx) * 2 / (self.fy * 0.85 * self.Column_depth)
            rho_steel = As_approx / (self.Column_width * self.Column_depth)

            # rho_steel=0.01  #get this from the chart
            Req_Area_of_steel = rho_steel * (self.Column_width * self.Column_depth)
            if verbose == 1:
                print('Req Area of steel: ', Req_Area_of_steel)
                print('rho_steel: ', rho_steel)
            if Req_Area_of_steel > 0:
                break

        # try different combinations of bar sizes:
        number_of_bars = np.zeros((len(db)))
        Area_of_steel = np.zeros((len(db)))
        M_cap = np.zeros((len(db)))
        for i in range(len(db)):
            req_num = Req_Area_of_steel / As_bar[i]
            trial_number = np.floor(req_num / 2) * 2
            print(trial_number)
            if float(trial_number) / req_num < 0.95:
                number_of_bars[i] = trial_number + 2
            else:
                number_of_bars[i] = trial_number
            if number_of_bars[i] == 6:
                number_of_bars[i] = 8
            Area_of_steel[i] = number_of_bars[i] * As_bar[i]

        if verbose == 1:
            print('Steel area: ', Area_of_steel)
        # Calculate the capacity:
        # set ep_c=0.003
        ep_c = 0.003
        Column_props = []
        for i in range(len(db)):
            if verbose == 1:
                print('bar size: ', db[i])
                print('number of bars', number_of_bars[i])
            if number_of_bars[i] < 8:
                M_cap[i] = 0
                Column_props.append([db[i], [], []])
            elif number_of_bars[i] > 22:
                M_cap[i] = 0
                Column_props.append([db[i], [], []])
            else:
                if number_of_bars[i] == 8:
                    AsArr = np.array([3, 2, 3])
                    dArr = np.array([0.85, 0.5, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 10:
                    AsArr = np.array([4, 2, 4])
                    dArr = np.array([0.85, 0.5, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 12:
                    AsArr = np.array([4, 2, 2, 4])
                    dArr = np.array([0.85, 0.65, 0.35, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 14:
                    AsArr = np.array([5, 2, 2, 5])
                    dArr = np.array([0.85, 0.65, 0.35, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 16:
                    AsArr = np.array([5, 2, 2, 2, 5])
                    dArr = np.array([0.85, 0.7, 0.5, 0.3, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 18:
                    AsArr = np.array([6, 2, 2, 2, 6])
                    dArr = np.array([0.85, 0.7, 0.5, 0.3, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 20:
                    AsArr = np.array([6, 2, 2, 2, 2, 6])
                    dArr = np.array([0.85, 0.75, 0.6, 0.4, 0.25, 0.15]) * self.Column_depth
                elif number_of_bars[i] == 22:
                    AsArr = np.array([6, 2, 2, 2, 2, 6])
                    dArr = np.array([0.85, 0.75, 0.6, 0.4, 0.25, 0.15]) * self.Column_depth

                Column_props.append([db[i], AsArr, dArr])
                # guess c:
                errC = 1.0
                fyArr = np.zeros(len(AsArr))
                for a_start in range(6):
                    if errC < 0.02:
                        break
                    c = (0.1 + 0.04 * a_start) * self.Column_depth
                    for attempt in range(100):
                        if errC < 0.02:
                            break
                        curve = ep_c / self.Column_depth
                        for  ele in range(len(AsArr)):
                            ep = (dArr[ele] - c) * curve
                            fyArr[ele] = min(abs(ep * self.E_s), self.fy) * np.sign(ep)
                        # print('fyArr: ',fyArr
                        c_new = (sum(AsArr * As_bar[i] * fyArr) + self.N_star) / \
                                    (alpha * beta * self.fc * self.Column_width)
                        errC = abs((c_new - c) / c)
                        c = c_new
                        # print('c: ',c

                if errC > 0.02:
                    M_cap[i] = 0
                else:
                    ConcC = alpha * beta * c * self.fc * self.Column_width
                    M_cap[i] = phi * (sum(AsArr * As_bar[i] * fyArr * (dArr - c)) + self.N_star * (
                                0.5 * self.Column_depth - c) - ConcC * (beta * c / 2 - c))
                # print('Conc moment: ',-ConcC*(beta*c/2-c)
                print('M_calculated: ', M_cap[i])
                if fyArr[0] == self.fy:
                    print('Tension failure')
                    compression_fail = 0
                else:
                    print('compression failure')
                    compression_fail = 1

        for i in range(len(db)):
            print('M_cap: ', M_cap[i])
            print('M_star: ', self.M_star)
            print('Column props: ', Column_props[i])
            if M_cap[i] != 0 and M_cap[i] > self.M_star / 2:
                self.Column_data = Column_props[i]
                self.Column_data.append(M_cap[i])
                print(self.Column_data)
                design_complete = 1
                break

        if design_complete == 0:
            print('Error with the design')
            print('M_cap: ', M_cap)
            print('Inputs:')
            print('M_star: ', self.M_star)
            print('N_star: ', self.N_star)
            print('Column_depth: ', self.Column_depth)
            print('Column_width: ', self.Column_width)
            print('fc: ', self.fc)
            print('fy: ', self.fy)
            print('E_s: ', self.E_s)
            raise ValueError()

        # print Column_data
        return [self.Column_data, design_complete]

    def plot_section(self, **kwargs):
        """
        This function plots the column cross-section based on the design from design()
        """
        show_plot = kwargs.get('show_plot', 1)
        save_on = kwargs.get('save_on', 1)
        self.SavePath = kwargs.get('save_path', self.SavePath)
        self.SectionName = kwargs.get('section_name', self.SectionName)

        # Draw reinforcing
        [db, AsArr, dArr, M_cap] = self.Column_data
        import matplotlib.pyplot as plt
        BIGFIG = plt.figure()
        sectfig = BIGFIG.add_subplot(111)
        rad = 361

        cover = 0.04
        spacing = (self.Column_width - 2 * cover - db) / (AsArr[0] - 1)
        bar_number = 0
        for layer in range(len(AsArr)):
            for bar in range(AsArr[layer]):
                bar_number += 1
                # print('inserting bar #: ',bar_number
                x_pos = cover + db / 2 + bar * spacing
                if bar == AsArr[layer] - 1:
                    x_pos = self.Column_width - db / 2 - cover
                circle1 = plt.Circle((x_pos, dArr[layer]), radius=db, color='k')
                blabel = str(AsArr[layer]) + '-D' + str(int(db * 1000))
                sectfig.add_patch(circle1)
                sectfig.text(self.Column_width + 0.05, dArr[layer] + 0.03, blabel)

        # Write moment capacity
        M_cap_str = 'Mn= \n' + str(float(int(M_cap / 100)) / 10) + 'kNm'
        sectfig.text(-0.15, self.Column_depth / 2 - self.Column_depth / 6 + self.Column_depth / 3, M_cap_str)

        # draw beam edge:
        x_edge = [0, self.Column_width, self.Column_width, 0, 0]
        y_edge = [0, 0, self.Column_depth, self.Column_depth, 0]
        edge = sectfig.plot(x_edge, y_edge)
        plt.setp(edge, c='k', linewidth=1.5)

        # Centring and scaling image
        plot_size = max(self.Column_width, self.Column_depth) + 0.1
        # print plot_size
        extra = plot_size - self.Column_width
        print('extra: ', extra)
        sectfig.axis('equal')
        sectfig.axis([-0.04 - extra / 2, self.Column_width + extra / 2 + 0.04, -0.01, plot_size + 0.01])
        sectfig.set_ylim([-0.04 - extra / 2, self.Column_depth + extra / 2 + 0.04])

        sectfig.set_xlabel('Width (m)')
        sectfig.set_ylabel('Depth (m)')
        sectfig.set_title(self.SectionName)

        if save_on == 1:
            figure_name = self.SavePath + self.SectionName + '.png'
            BIGFIG.savefig(figure_name, format='png')
        if show_plot == 1:
            plt.show()

        del BIGFIG
        plt.clf()
        plt.close()


def run_example():
    moment = 250.0e3
    axial = 150.0e3
    Column_length = 3.2
    depth = 0.5
    width = 0.4
    fc = 30e6
    fy_col = 300e6
    e_mod_steel = 200.0e9
    min_column_depth = 0.5
    preferred_bar_diam = 0.02
    preferred_cover = 0.04
    layer_spacing = 0.04

    columnsection = ColumnSection(moment, axial, depth, width, fc, fy_col,
                                  e_mod_steel, preferred_bar_diam)
    columnsection.design()
    columnsection.plot_section()

if __name__ == '__main__':
    run_example()