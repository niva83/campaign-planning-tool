import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os, shutil

class Plot():
    def plot_layer(self, **kwargs):
        """
        Plots individual GIS layers.
    
        Parameters
        ----------
        **kwargs : see below
    
        Keyword Arguments
        -----------------
        layer_id : str
            A string indicating which layer to be plotted.        
        title : str
            The plot title.
        legend_label : str
            The legend label indicating what parameter is plotted.
        save_plot : bool
            Indicating whether to save the plot as PDF.
        input_type : str
    
    
        Returns
        -------
        plot : matplotlib
    
        """
        if 'layer_id' in kwargs:
            if kwargs['layer_id'] in self.layer_creation_info:
                layer = self.layer_selector(kwargs['layer_id'])
                layer_info = self.layer_creation_info[kwargs['layer_id']]

                # Assuring there is something to plot!
                if layer is not None:
                    # Assuring there is more than one type of value in layer!
                    if len(np.unique(layer)) > 1:
                        # Preparing resolution for pcolormesh
                        min_value = np.min(layer)
                        max_value = np.max(layer)

                        if len(layer.shape) > 2:
                            # this is for plottin layers which dimensions is
                            # (ncols, nrows, len(measurement_pts))
                            layer = np.sum(layer, axis = 2)
                            levels = np.array(range(-1,int(np.max(layer)) + 1, 1))
                            boundaries = levels + 0.5
                        else:
                            # this is for plottin layers which dimensions is
                            # (ncols, nrows) such as for example orography
                            increment = abs(max_value - min_value)/20
                            levels = np.linspace(min_value, max_value, 20)
                            boundaries = np.linspace(min_value - increment/2, 
                                                    max_value + increment/2, 21)          

                
                        fig, ax = plt.subplots(sharey = True, 
                                               figsize=(800/self.MY_DPI, 
                                                        800/self.MY_DPI), 
                                               dpi=self.MY_DPI)

                        cmap = plt.cm.RdBu_r
                        cs = plt.pcolormesh(self.x, self.y, 
                                            layer, cmap=cmap, alpha = 1)
                    
                        # values for 'fraction' and 'pad' are manually tuned
                        # most likely depend on the chose resolution of the plot!
                        cbar = plt.colorbar(cs,orientation='vertical', 
                                            ticks=levels, 
                                            boundaries=boundaries,
                                            fraction=0.047, pad=0.01)


                        if 'legend_label' in kwargs:
                            cbar.set_label(kwargs['legend_label'], 
                                        fontsize = self.FONT_SIZE)
                        else:
                            cbar.set_label(self.legend_label, 
                                        fontsize = self.FONT_SIZE)


                        points = None
                        if layer_info['points_id'] is not None:
                            points = self.measurement_type_selector(
                                                       layer_info['points_id'])
                            
                    
                        if points is not None:
                            for i, pts in enumerate(points):
                                if i == 0:
                                    ax.scatter(pts[0], pts[1], marker='o', 
                                               facecolors='yellow', 
                                               edgecolors='black', 
                                               s=80, zorder=1500, 
                                               label = 'points: ' 
                                                     + layer_info['points_id'])                    
                                else:
                                    ax.scatter(pts[0], pts[1], marker='o',
                                               facecolors='yellow', 
                                               edgecolors='black', 
                                               s=80,zorder=1500)
                    
                        if layer_info['lidars_id'] is not None:
                            lidar_info = self.lidar_dictionary[
                                                       layer_info['lidars_id']]

                            
                            ax.scatter(lidar_info['position'][0], 
                                       lidar_info['position'][1],
                                       marker='s', 
                                       facecolors=self.COLOR_LIST[1], 
                                       edgecolors='white', linewidth='2',
                                       s=100, zorder=2000, 
                                       label = 'lidar: ' 
                                               + layer_info['lidars_id'])

                            visible_pts = points[np.where(
                                             lidar_info['reachable_points']>0)]

                            for i in range(0,len(visible_pts)):
                                if i == 0:
                                    ax.scatter(visible_pts[i][0], 
                                               visible_pts[i][1], 
                                               marker='+', color='black', 
                                               s=80, zorder=2000, 
                                               label = "points: reachable")
                                else:
                                    ax.scatter(visible_pts[i][0], 
                                               visible_pts[i][1], 
                                               marker='+', color='black', 
                                               s=80, zorder=2000)

                        plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
                        plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
                    
                        if 'title' in kwargs:
                            plt.title(kwargs['title'], fontsize = self.FONT_SIZE)
                    
                        ax.set_aspect(1.0)
                        ax.legend(loc='lower right', fontsize = self.FONT_SIZE)        
                        plt.show()
                    
                        if ('title' in kwargs 
                            and 'save_plot' in kwargs 
                            and kwargs['save_plot']):
                            file_name_str = kwargs['title'] + ".pdf"
                            file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                            fig.savefig(file_path.absolute().as_posix(), 
                                        bbox_inches='tight')
                #     else:
                #         print('Provided layer elements all have the same value equal to: ' + str(np.unique(layer)[0]))
                # else:
                #     print('Provided layer does not exist!')

    def plot_optimization(self, **kwargs):
        """
        Plots measurement point optimization result.
        
        Parameters
        ----------
        **kwargs : see below

        Keyword Arguments
        -----------------
        save_plot : bool
            Indicating whether to save the plot as PDF.

        See also
        --------
        optimize_measurements : implementation of disc covering problem
        add_measurements : method for adding initial measurement points

        Notes
        -----
        To generate the plot it is required that 
        the measurement optimization was performed.

        Returns
        -------
        plot : matplotlib
        
        """
        if 'points_id' in kwargs and kwargs['points_id'] in self.POINTS_TYPE:
            measurement_pts = self.measurement_type_selector(kwargs['points_id'])
            pts_str = kwargs['points_id']
        else:
            measurement_pts = self.measurement_type_selector('initial')
            pts_str = 'initial'


        if measurement_pts is not None and self.measurement_type_selector('optimized') is not None:
            fig, ax = plt.subplots(sharey = True, figsize=(800/self.MY_DPI, 800/self.MY_DPI), dpi=self.MY_DPI)

            for i,pt in enumerate(measurement_pts):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='red', edgecolors='black', 
                        s=10,zorder=1500, label = "points: " + pts_str)
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                                        facecolors='red', edgecolors='black', 
                                        s=10,zorder=1500,)            


            for i,pt in enumerate(self.measurement_type_selector('optimized') ):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='white', edgecolors='black', 
                        s=10,zorder=1500, label = "points: optimized")
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', edgecolor='black', 
                                            zorder=0,  alpha = 0.5))                 
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='white', edgecolors='black', 
                        s=10,zorder=1500)
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', edgecolor='black', 
                                            zorder=0,  alpha = 0.5))                 
    
                    

            plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
            plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
            ax.legend(loc='lower right', fontsize = self.FONT_SIZE)


            ax.set_xlim(np.min(self.x),np.max(self.x))
            ax.set_ylim(np.min(self.y),np.max(self.y))

            ax.set_aspect(1.0)
            plt.show()
            if 'save_plot' in kwargs and kwargs['save_plot']:
                file_name_str ="measurements_optimized.pdf"
                file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                fig.savefig(file_path.absolute().as_posix(), bbox_inches='tight')

    # change so it shows what you get when you
    # intersect info from one or more lidars
    # but without a trajectory ...
    def plot_layout(self, **kwargs):
        """
        Plots campaign layout.
        
        Parameters
        ----------
        layer : ndarray
            nD array containing data with `float` or `int` type 
            corresponding to a specific GIS layer.
        **kwargs : see below

        Keyword Arguments
        -----------------
        title : str
            The plot title.
        legend_label : str
            The legend label indicating what parameter is plotted.
        save_plot : bool
            Indicating whether to save the plot as PDF.
        input_type : str

        
        Returns
        -------
        plot : matplotlib
        
        Examples
        --------
        >>> layout.plot_GIS_layer(layout.orography_layer, levels = np.array(range(0,510,10)), title = 'Orography', legend_label = 'Height asl [m]' , save_plot = True)

        """
        if self.flags['trajectory_optimized']:

            # levels = np.array(range(-1,self.combined_layer.shape[-1] + 1, 1))
            # layer = np.sum(self.combined_layer, axis = 2)
            levels = np.linspace(np.min(self.orography_layer), np.max(self.orography_layer), 20)

            fig, ax = plt.subplots(sharey = True, figsize=(800/self.MY_DPI, 800/self.MY_DPI), dpi=self.MY_DPI)
            cmap = plt.cm.Greys
            cs = plt.contourf(self.x, self.y, self.orography_layer, levels=levels, cmap=cmap, alpha = 0.4)

            cbar = plt.colorbar(cs,orientation='vertical',fraction=0.047, pad=0.01)
            cbar.set_label('Height asl [m]', fontsize = self.FONT_SIZE)
            
            ax.scatter(self.lidar_pos_1[0], self.lidar_pos_1[1], marker='o', 
            facecolors='black', edgecolors='white', s=60, zorder=2000, label = "lidar_1")

            ax.scatter(self.lidar_pos_2[0], self.lidar_pos_2[1], marker = 'o', 
            facecolors='white', edgecolors='black',s=60,zorder=2000, label = "lidar_2")

            if 'points_id' in kwargs and kwargs['points_id'] in self.POINTS_TYPE:
                measurement_pts = self.measurement_type_selector(kwargs['points_id'])
            else:
                measurement_pts = self.measurement_type_selector(self.measurements_selector)        

            if measurement_pts is not None:
                for i, pts in enumerate(measurement_pts):
                    if i == 0:
                        ax.scatter(pts[0], pts[1], marker='o', 
                        facecolors='yellow', edgecolors='black', 
                        s=60,zorder=1500, label = 'measurements_' + self.measurements_selector)                    
                    else:
                        ax.scatter(pts[0], pts[1], marker='o',
                        facecolors='yellow', edgecolors='black', 
                        s=60,zorder=1500)

            if self.measurements_reachable is not None:
                for i in range(0,len(self.measurements_reachable)):
                    if i == 0:
                        ax.scatter(self.measurements_reachable[i][0], self.measurements_reachable[i][1], 
                                marker='+', 
                                color='red',  edgecolors='black', 
                                s=80,zorder=2000, label = "measurements_reachable")
                    else:
                        ax.scatter(self.measurements_reachable[i][0], self.measurements_reachable[i][1], 
                                marker='+', 
                                facecolors='red', 
                                s=80,zorder=2000)

                ax.plot(self.trajectory[:,0],self.trajectory[:,1],
                        color='blue', linestyle='--',linewidth=1, zorder=3000,label='trajectory')

                ax.scatter(self.trajectory[0,0], self.trajectory[0,1], 
                       marker='o', facecolors='white', edgecolors='green',s=120,zorder=1400,label = "trajectory start")



            ax.legend(loc='lower right', fontsize = self.FONT_SIZE)    

            plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
            plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)


            plt.title('Campaign layout', fontsize = self.FONT_SIZE)

            ax.set_aspect(1.0)
            plt.show()

            if 'save_plot' in kwargs and kwargs['save_plot']:
                file_name_str ="campaign_design.pdf"
                file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                fig.savefig(file_path.absolute().as_posix(), bbox_inches='tight')
        else:
            print('Trajectory not optimized -> nothing to plot')


