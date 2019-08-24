import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os, shutil

class Plot():
    """
    A class containing methods for plotting CPT results.

    Methods
    -------
    plot_layer(layer_id, **kwargs)
        Plots individual GIS layers an optionally lidar positions.
    plot_optimization(**kwargs)
        Plots measurement point optimization result.
    plot_design(layer_id, lidar_ids, **kwargs)
        Plots measurement point optimization result.        

    """
    __COLOR_LIST = ['blue', 'green', 'red', 'purple', 
                  'brown', 'pink', 'gray', 'olive', 'cyan']

    def plot_layer(self,layer_id, **kwargs):
        """
        Plots individual GIS layers and lidar positions.
    
        Parameters
        ----------
        layer_id : str, required
            A string indicating which layer to be plotted.

    
        Keyword Arguments
        -----------------
        lidar_ids : list of str, optional
            A list of strings indicating which lidars should be plotted.        
        title : str, optional
            The plot title.
        legend_label : str, optional
            The legend label indicating what parameter is plotted.
        save_plot : bool, optional
            Indicating whether to save the plot as PDF.

        Returns
        -------
        plot : matplotlib
    
        """

        if layer_id in self.layers_info:
            layer = self.layer_selector(layer_id)
            layer_info = self.layers_info[layer_id]

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
                    # made sure the plot pixel is centered at position
                    # not position respect to the edge.
                    cs = plt.pcolormesh(self.x - self.MESH_RES/2, 
                                        self.y- self.MESH_RES/2, 
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
                        points = self.points_selector(
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

                    if ('lidar_ids' in kwargs and 
                            set(kwargs['lidar_ids']).issubset(self.lidar_dictionary)):

                        for i, lidar in enumerate(kwargs['lidar_ids']):
                            lidar_info = self.lidar_dictionary[lidar]
                            if lidar_info['inside_mesh']:
                                ax.scatter(lidar_info['position'][0], 
                                        lidar_info['position'][1],
                                        marker='s', 
                                        facecolors=self.__COLOR_LIST[i], 
                                        edgecolors='white', linewidth='2',
                                        s=100, zorder=2000, 
                                        label = 'lidar: ' 
                                                + lidar)
                                visible_pts = points[np.where(
                                                lidar_info['reachable_points']>0)]

                                for j in range(0,len(visible_pts)):
                                    if j == 0:
                                        ax.scatter(visible_pts[j][0], 
                                                visible_pts[j][1], 
                                                marker='o', 
                                                facecolors=self.__COLOR_LIST[i], 
                                                edgecolors='white',
                                                s=80 + 200 *( i*2 + 1), 
                                                zorder=1500 - 20*( i + 1), 
                                                label = "reachable by " + lidar)
                                    else:
                                        ax.scatter(visible_pts[j][0], 
                                                visible_pts[j][1], 
                                                marker='o', 
                                                facecolors=self.__COLOR_LIST[i], 
                                                edgecolors='white', 
                                                                                                        
                                                s=80 + 200 *( i*2 + 1), 
                                                zorder=1500 - 20*( i + 1))

                                


                    plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
                    plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
                
                    if 'title' in kwargs:
                        plt.title(kwargs['title'], fontsize = self.FONT_SIZE)
                
                    ax.set_aspect(1.0)
                    legend = ax.legend(loc='lower right', 
                                        fontsize = self.FONT_SIZE, 
                                        bbox_to_anchor=(1.6, 0))
                    
                    for handle in legend.legendHandles:
                        handle.set_sizes([80])
                    plt.show()
                
                    if 'save_plot' in kwargs and kwargs['save_plot']:
                        file_name_str = layer_id + ".pdf"
                        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                        fig.savefig(file_path.absolute().as_posix(), 
                                    bbox_inches='tight')

    def plot_optimization(self, **kwargs):
        """
        Plots measurement point optimization result.
        
        Parameters
        ----------
        **kwargs : see below

        Keyword Arguments
        -----------------
        save_plot : bool, optional
            Indicating whether to save the plot as PDF.

        See also
        --------
        optimize_measurements : implementation of disc covering problem

        Notes
        -----
        To generate the plot it is required that 
        the measurement optimization was performed.

        Returns
        -------
        plot : matplotlib
        
        """
        if 'points_id' in kwargs and kwargs['points_id'] in self.POINTS_ID:
            measurement_pts = self.points_selector(kwargs['points_id'])
            pts_str = kwargs['points_id']
        else:
            measurement_pts = self.points_selector('initial')
            pts_str = 'initial'


        if (measurement_pts is not None 
            and self.points_selector('optimized') is not None):
            fig, ax = plt.subplots(sharey = True, 
                                   figsize=(800/self.MY_DPI, 800/self.MY_DPI), 
                                   dpi=self.MY_DPI)

            for i,pt in enumerate(measurement_pts):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', 
                        facecolors='red', edgecolors='black', 
                        s=10,zorder=1500, label = "points: " + pts_str)
                else:
                    ax.scatter(pt[0], pt[1],marker='o', 
                                        facecolors='red', edgecolors='black', 
                                        s=10,zorder=1500,)            


            for i,pt in enumerate(self.points_selector('optimized')):
                if i == 0:
                    ax.scatter(pt[0], pt[1],marker='o', facecolors='white', 
                               edgecolors='black', s=10,zorder=1500, 
                               label = "points: optimized")
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', 
                                            edgecolor='black', 
                                            zorder=0,  alpha = 0.5))                 
                else:
                    ax.scatter(pt[0], pt[1],marker='o', facecolors='white', 
                               edgecolors='black', s=10,zorder=1500)
                    ax.add_artist(plt.Circle((pt[0], pt[1]), 
                                            self.REP_RADIUS,                               
                                            facecolor='grey', 
                                            edgecolor='black', 
                                            zorder=0,  alpha = 0.5))                 
    
                    

            plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
            plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)



            ax.set_xlim(np.min(self.x),np.max(self.x))
            ax.set_ylim(np.min(self.y),np.max(self.y))

            ax.set_aspect(1.0)
            legend = ax.legend(loc='lower right', 
                               fontsize = self.FONT_SIZE, 
                               bbox_to_anchor=(1.6, 0))
            
            for handle in legend.legendHandles:
                handle.set_sizes([80])
            plt.show()
            if 'save_plot' in kwargs and kwargs['save_plot']:
                file_name_str ="measurements_optimized.pdf"
                file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                fig.savefig(file_path.absolute().as_posix(), 
                            bbox_inches='tight')




    def plot_design(self, layer_id, lidar_ids, **kwargs):
        """
        Plots campaign design.
        
        Parameters
        ----------
        layer_id : str, required
            A string indicating which layer to be plotted.
        lidar_ids : list of str, required
            A list of strings indicating which lidars should be plotted.        

        Keyword Arguments
        -----------------
        title : str, optional
            The plot title.
        legend_label : str, optional
            The legend label indicating what parameter is plotted.
        save_plot : bool, optional
            Indicating whether to save the plot as PDF.

        
        Returns
        -------
        plot : matplotlib
        
        """

        if layer_id in self.layers_info and self.flags['trajectory_optimized']:
            layer = self.layer_selector(layer_id)
            layer_info = self.layers_info[layer_id]

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
                    cs = plt.pcolormesh(self.x - self.MESH_RES/2, 
                                        self.y - self.MESH_RES/2, 
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
                        points = self.points_selector(
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

                    if set(lidar_ids).issubset(self.lidar_dictionary):
                        trajectory_plotted = False

                        for i, lidar in enumerate(lidar_ids):
                            lidar_info = self.lidar_dictionary[lidar]
                            if lidar_info['inside_mesh']:
                                ax.scatter(lidar_info['position'][0], 
                                        lidar_info['position'][1],
                                        marker='s', 
                                        facecolors=self.__COLOR_LIST[i], 
                                        edgecolors='white', linewidth='2',
                                        s=100, zorder=2000, 
                                        label = 'lidar: ' 
                                                + lidar)
                                if trajectory_plotted == False:
                                    trajectory_plotted == True
                                    trajectory = lidar_info['trajectory'].values[:, 1:]
                                    ax.plot(trajectory[:,0],trajectory[:,1],
                                            color='black', 
                                            linestyle='--',
                                            linewidth=2, 
                                            zorder=3000,
                                            label='trajectory')
                                    
                                    ax.plot(trajectory[:,0],trajectory[:,1],
                                            color='white', 
                                            linestyle='--',
                                            linewidth=4, 
                                            zorder=2500)
                                    
                                    for j in range(0,len(trajectory)):
                                        if j == 0:
                                            ax.scatter(trajectory[j][0], 
                                                    trajectory[j][1], 
                                                    marker='o', 
                                                    facecolors='white', 
                                                    edgecolors='black',
                                                    s=80 + 200 *( i*2 + 1), 
                                                    zorder=1500 - 20*( i + 1), 
                                                    label = "points: trajectory")
                                            ax.scatter(trajectory[j][0], 
                                                    trajectory[j][1], 
                                                    marker='o', 
                                                    facecolors='green', 
                                                    edgecolors='white',
                                                    s=80 + 200 *( i*2 + 2), 
                                                    zorder=1500 - 20*( i + 2), 
                                                    label = "trajectory start")

                                        else:
                                            ax.scatter(trajectory[j][0], 
                                                    trajectory[j][1], 
                                                    marker='o', 
                                                    facecolors='white', 
                                                    edgecolors='black', 

                                                    s=80 + 200 *( i*2 + 1), 
                                                    zorder=1500 - 20*( i + 1))
                                    trajectory_plotted = True

                                


                    plt.xlabel('Easting [m]', fontsize = self.FONT_SIZE)
                    plt.ylabel('Northing [m]', fontsize = self.FONT_SIZE)
                
                    if 'title' in kwargs:
                        plt.title(kwargs['title'], fontsize = self.FONT_SIZE)
                
                    ax.set_aspect(1.0)
                    legend = ax.legend(loc='lower right', 
                                        fontsize = self.FONT_SIZE, 
                                        bbox_to_anchor=(1.6, 0))
                    
                    for handle in legend.legendHandles:
                        if hasattr(handle, 'set_sizes'):
                            handle.set_sizes([80])
                    plt.show()
                
                    if 'save_plot' in kwargs and kwargs['save_plot']:
                        file_name_str = "campaign_design.pdf"
                        file_path = self.OUTPUT_DATA_PATH.joinpath(file_name_str) 
                        fig.savefig(file_path.absolute().as_posix(), 
                                    bbox_inches='tight')