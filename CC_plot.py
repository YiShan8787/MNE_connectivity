import time

import numpy as np
import pyqtgraph.opengl as gl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy import linalg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from channel_loca_dict import channel_dict_3D
from ws_CC import WS_CC

class CC_Plot(QtGui.QWidget):
    resized = QtCore.pyqtSignal()
    def __init__(self, url=None):
        super().__init__()
        
        if url is None:
            ws_url = "ws://localhost:7777"
        else:
            ws_url = url
        
        self.ws_CC = WS_CC(CC_plot=self, url=ws_url)
        self.method = "pli"
        self.colorbar_label = "Phase Lag Index (PLI)"
        self.mode = "multitaper"
        self.event_id = None
        self.fmin = 0.
        self.fmax = 30.
        self.pos = list(channel_dict_3D.values())
        self.ch_names_ = list(channel_dict_3D.keys())
        self.pre_line = list()
        self.init_ui()
        self.timer_interval = 0.1
        self.time_scale = 10
        self.setup_signal_handler()
        self.show()
        
    def init_ui(self):
        self.setWindowTitle("Connectivity")
        self.resize(1000, 600)
        grid = QtGui.QGridLayout(self)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 9)
        grid.addWidget(self.parameter_group(), 0, 0, 1, 1)
        #grid.addWidget(self.message_group(), 1, 0, 1, 1)
        grid.addWidget(self.plot_group(), 0, 1, 1, 1)
        
    def parameter_group(self):
        groupBox = QtGui.QGroupBox("Parameter")
        
        method_comboBox = QtWidgets.QComboBox()
        method_comboBox.addItems(["pli", "wpli", "coh", "imcoh", "plv"])
        method_comboBox.activated[str].connect(self.CC_method_handler)
        _method_comboBox = QtWidgets.QLabel()
        _method_comboBox.setText("method:")
        
        mode_comboBox = QtWidgets.QComboBox()
        mode_comboBox.addItems(["multitaper", "fourier"])
        mode_comboBox.activated[str].connect(self.CC_mode_handler)
        _mode_comboBox = QtWidgets.QLabel()
        _mode_comboBox.setText("mode:")
        
        
        freq_range_comboBox = QtWidgets.QComboBox()
        freq_range_comboBox.addItems(["default(0-50)", "Delta", "Theta", "Alpha", "Beta", "Gamma"])
        freq_range_comboBox.activated[str].connect(self.CC_freq_range_handler)
        _freq_range_comboBox = QtWidgets.QLabel()
        _freq_range_comboBox.setText("freq_range:")
        
        self.event_id_comboBox = QtWidgets.QComboBox()
        self.event_id_comboBox.addItem("None")
        self.event_id_comboBox.activated[str].connect(self.CC_event_id_handler)
        _event_id_comboBox = QtWidgets.QLabel()
        _event_id_comboBox.setText("event_id")
        
        scalp_mesh_checkBox = QtWidgets.QCheckBox()
        scalp_mesh_checkBox.setChecked(True)
        scalp_mesh_checkBox.stateChanged.connect(self.CC_scalp_mesh_handler)
        _scalp_mesh_checkBox = QtWidgets.QLabel()
        _scalp_mesh_checkBox.setText("scalp:")
        
        skull_mesh_checkBox = QtWidgets.QCheckBox()
        skull_mesh_checkBox.setChecked(True)
        skull_mesh_checkBox.stateChanged.connect(self.CC_skull_mesh_handler)
        _skull_mesh_checkBox = QtWidgets.QLabel()
        _skull_mesh_checkBox.setText("skull:")
        
        csf_mesh_checkBox = QtWidgets.QCheckBox()
        csf_mesh_checkBox.setChecked(True)
        csf_mesh_checkBox.stateChanged.connect(self.CC_csf_mesh_handler)
        _csf_mesh_checkBox = QtWidgets.QLabel()
        _csf_mesh_checkBox.setText("csf:")
        
        LONI_mesh_checkBox = QtWidgets.QCheckBox()
        LONI_mesh_checkBox.setChecked(True)
        LONI_mesh_checkBox.stateChanged.connect(self.CC_LONI_mesh_handler)
        _LONI_mesh_checkBox = QtWidgets.QLabel()
        _LONI_mesh_checkBox.setText("LONI:")
        
        self.LONI_mesh_color_checkBox = QtWidgets.QCheckBox()
        self.LONI_mesh_color_checkBox.setChecked(True)
        self.LONI_mesh_color_checkBox.stateChanged.connect(self.CC_LONI_mesh_color_handler)
        _LONI_mesh_color_checkBox = QtWidgets.QLabel()
        _LONI_mesh_color_checkBox.setText("LONI_color:")
        
        gridlayout = QtGui.QGridLayout()
        gridlayout.addWidget(_freq_range_comboBox, 0, 0, 1, 1)
        gridlayout.addWidget(freq_range_comboBox, 0, 1, 1, 1)
        gridlayout.addWidget(_method_comboBox, 1, 0, 1, 1)
        gridlayout.addWidget(method_comboBox, 1, 1, 1, 1)
        gridlayout.addWidget(_mode_comboBox, 2, 0, 1, 1)
        gridlayout.addWidget(mode_comboBox, 2, 1, 1, 1)
        gridlayout.addWidget(_event_id_comboBox, 3, 0, 1, 1)
        gridlayout.addWidget(self.event_id_comboBox, 3, 1, 1, 1)
        gridlayout.addWidget(_scalp_mesh_checkBox, 4, 0, 1, 1)
        gridlayout.addWidget(scalp_mesh_checkBox, 4, 1, 1, 1)
        gridlayout.addWidget(_skull_mesh_checkBox, 5, 0, 1, 1)
        gridlayout.addWidget(skull_mesh_checkBox, 5, 1, 1, 1)
        gridlayout.addWidget(_csf_mesh_checkBox, 6, 0, 1, 1)
        gridlayout.addWidget(csf_mesh_checkBox, 6, 1, 1, 1)
        gridlayout.addWidget(_LONI_mesh_checkBox, 7, 0, 1, 1)
        gridlayout.addWidget(LONI_mesh_checkBox, 7, 1, 1, 1)
        gridlayout.addWidget(_LONI_mesh_color_checkBox, 8, 0, 1, 1)
        gridlayout.addWidget(self.LONI_mesh_color_checkBox, 8, 1, 1, 1)
        groupBox.setLayout(gridlayout)
        
        return groupBox
    
    def message_group(self):
        groupBox = QtGui.QGroupBox("Message")
        
        self.log_widget = QtGui.QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        #self.print_system_log("In Connectivity")
        
        gridlayout = QtGui.QGridLayout()
        gridlayout.addWidget(self.log_widget, 0, 0)
        groupBox.setLayout(gridlayout)
        
        return groupBox
        
    def print_system_log(self, message):
        if type(message) == str:
            self.log_widget.appendPlainText(time.strftime("%H:%M:%S\t")+message)
            
    def plot_group(self):
        groupBox = QtGui.QGroupBox("3D Plot")
        
        self.glWidget = gl.GLViewWidget()
        self.glWidget.opts['distance'] = 24
        
        self.ch_loc_3D_idx = [self.ch_names_.index(name) for name in self.ws_CC.ch_label]
        self.ch_loc_3D = np.array([self.pos[idx] for idx in self.ch_loc_3D_idx])
        self.ch_loc_3D_idx.sort(reverse=True)
        for idx in self.ch_loc_3D_idx:
            del self.pos[idx]
            
        pos = np.array(self.pos)
        pos *= 9
        sp1 = gl.GLScatterPlotItem(pos=pos, color=(1,0,0,10.7))
        self.glWidget.addItem(sp1)
        
        self.ch_loc_3D *= 9
        sp2 = gl.GLScatterPlotItem(pos=self.ch_loc_3D, color=(0,0,1,10.5))
        self.glWidget.addItem(sp2)
        
        label = MyLabelItem(pos=self.ch_loc_3D, text=self.ws_CC.ch_label, glWidget=self.glWidget)
        self.glWidget.addItem(label)
        
        #load scalp
        verts = self.objLoader('scalp_bem_mesh.txt')
        faces = self.objLoader('scalp_bem_mesh_face.txt')
        verts = np.array(verts,dtype='float32')
        faces = np.array(faces,dtype='int32')
        self.scalp_bem_mesh = gl.GLMeshItem(vertexes=verts ,faces=faces ,smooth=False,drawEdges = False)
        self.scalp_bem_mesh.translate(0, 1.9, 0)       #改變整個ml的中心位置，ex:(1.3,2.5,-1.2),(1.5,2,0)
        self.scalp_bem_mesh.scale(0.13, 0.1, 0.103)        #改變整個ml在各個方向上的比例，ex:(4.1,4.1,4.1)(3.8,3.8,3.8)
        self.scalp_bem_mesh.opts['color'] = (1.,1.,1.,0.06)
        self.scalp_bem_mesh.setGLOptions('additive')          #透明度啟用
        self.glWidget.addItem(self.scalp_bem_mesh)
        #load skull
        verts = self.objLoader('skull_bem_mesh.txt')
        faces = self.objLoader('skull_bem_mesh_face.txt')
        verts = np.array(verts,dtype='float32')
        faces = np.array(faces,dtype='int32')
        self.skull_bem_mesh = gl.GLMeshItem(vertexes=verts ,faces=faces ,smooth=False,drawEdges = False)
        self.skull_bem_mesh.translate(0, 1.9, 0)       #改變整個ml的中心位置，ex:(1.3,2.5,-1.2),(1.5,2,0)(0, 2.5, 0)
        self.skull_bem_mesh.scale(0.13, 0.1, 0.103)        #改變整個ml在各個方向上的比例，ex:(4.1,4.1,4.1)(3.8,3.8,3.8)(0.13, 0.11, 0.11)
        self.skull_bem_mesh.opts['color'] = (1.,1.,1.,0.05)
        self.skull_bem_mesh.setGLOptions('additive')          #透明度啟用
        self.glWidget.addItem(self.skull_bem_mesh)
        #load csf
        verts = self.objLoader('csf_bem_mesh.txt')
        faces = self.objLoader('csf_bem_mesh_face.txt')
        verts = np.array(verts,dtype='float32')
        faces = np.array(faces,dtype='int32')
        self.csf_bem_mesh = gl.GLMeshItem(vertexes=verts ,faces=faces ,smooth=False,drawEdges = False)
        self.csf_bem_mesh.translate(0, 1.9, 0)       #改變整個ml的中心位置，ex:(1.3,2.5,-1.2),(1.5,2,0)
        self.csf_bem_mesh.scale(0.13, 0.1, 0.103)        #改變整個ml在各個方向上的比例，ex:(4.1,4.1,4.1)(3.8,3.8,3.8)
        self.csf_bem_mesh.opts['color'] = (1.,1.,1.,0.04)
        self.csf_bem_mesh.setGLOptions('additive')          #透明度啟用
        self.glWidget.addItem(self.csf_bem_mesh)
        #loas LONI
        verts = self.objLoader('LONI_mesh.txt')
        faces = self.objLoader('LONI_mesh_face.txt')
        colors = self.objLoader('LONI_mesh_color.txt')
        labels = self.labelLoader('LONI_mesh_label.txt')
        self.verts = np.array(verts,dtype='float32')
        self.faces = np.array(faces,dtype='int32')
        colors = np.array(colors,dtype='float32')
        labels = np.array(labels,dtype='float32')
        self.vertex_cols = np.ones((self.verts.shape[0], 4), dtype='float32')
        for i in range(colors.shape[0]):
            index = labels==colors[i, 4]
            self.vertex_cols[index,:3] = colors[i,:3]
            self.vertex_cols[index,3] = 0.3
        self.LONI_mesh = gl.GLMeshItem(vertexes=self.verts, faces=self.faces, vertexColors=self.vertex_cols, smooth=False, drawEdges = False)
        self.LONI_mesh.translate(0, 1.9, 0)       #改變整個ml的中心位置，ex:(1.3,2.5,-1.2),(1.5,2,0)
        self.LONI_mesh.scale(0.13, 0.1, 0.103)        #改變整個ml在各個方向上的比例，ex:(4.1,4.1,4.1)(3.8,3.8,3.8)
        self.LONI_mesh.opts['color'] = (1.,1.,1.,0.02)
        self.LONI_mesh.setGLOptions('additive')          #透明度啟用
        self.glWidget.addItem(self.LONI_mesh)
        
        
        figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(figure)
        self.ax = figure.add_axes([0.05, 0.40, 0.9, 0.23])
        self.col_map = plt.get_cmap('bwr')
        self.norm = mpl.colors.Normalize(vmin=0., vmax=1.)
        self.colorbar = mpl.colorbar.ColorbarBase(self.ax, cmap=self.col_map, norm=self.norm, orientation = 'horizontal')
                                
        gridlayout = QtGui.QGridLayout()
        gridlayout.setRowStretch(0, 15)
        gridlayout.setRowStretch(1, 2)
        gridlayout.addWidget(self.glWidget, 0, 0)
        gridlayout.addWidget(self.canvas, 1, 0)
        groupBox.setLayout(gridlayout)
        
        return groupBox
        
    def objLoader(self, file):
        lst = list()
        for line in open(file, 'r'):
            line = line.split()
            lst.append(line)
        return lst
    
    def labelLoader(self, file):
        lst = list()
        for line in open(file, 'r'):
            lst.append(line)
        return lst
    
    def CC_method_handler(self, text):
        self.method = text
        #["pli", "wpli", "coh", "cohy", "plv"]
        if text == "pli":
            self.colorbar_label = "Phase Lag Index (PLI)"
        elif text == "wpli":
            self.colorbar_label = "Weighted Phase Lag Index (WPLI)"
        elif text == "coh":
            self.colorbar_label = "Coherence"
        elif text == "plv":
            self.colorbar_label = "Phase-Locking Value (PLV)"
        
    def CC_mode_handler(self, text):
        self.mode = text
        
    def CC_freq_range_handler(self, text):
        if text == "default(0-50)":
            self.fmin = 0
            self.fmax = 50
        elif text == "Delta":
            self.fmin = 0.1
            self.fmax = 3
        elif text == "Theta":
            self.fmin = 4
            self.fmax = 7
        elif text == "Alpha":
            self.fmin = 8
            self.fmax = 14
        elif text == "Beta":
            self.fmin = 12.5
            self.fmax = 28
        elif text == "Gamma":
            self.fmin = 25
            self.fmax = 50
        #self.print_system_log("freq_range:%s fmax=%d fmin=%d" % (text, self.fmax, self.fmin))
        
    def CC_event_id_handler(self, text):
        if text == "None":
            self.event_id = None
        else:
            self.event_id = int(text)
            
    def CC_scalp_mesh_handler(self, state):
        if state == 0:
            self.glWidget.removeItem(self.scalp_bem_mesh)
        else:
            self.glWidget.addItem(self.scalp_bem_mesh)
    
    def CC_skull_mesh_handler(self, state):
        if state == 0:
            self.glWidget.removeItem(self.skull_bem_mesh)
        else:
            self.glWidget.addItem(self.skull_bem_mesh)
    
    def CC_csf_mesh_handler(self, state):
        if state == 0:
            self.glWidget.removeItem(self.csf_bem_mesh)
        else:
            self.glWidget.addItem(self.csf_bem_mesh)
    
    def CC_LONI_mesh_handler(self, state):
        if state == 0:
            self.glWidget.removeItem(self.LONI_mesh)
            self.LONI_mesh_color_checkBox.setEnabled(False)
        else:
            self.glWidget.addItem(self.LONI_mesh)
            self.LONI_mesh_color_checkBox.setEnabled(True)
    
    def CC_LONI_mesh_color_handler(self, state):
        if state == 0:
            self.glWidget.removeItem(self.LONI_mesh)
            self.LONI_mesh = gl.GLMeshItem(vertexes=self.verts, faces=self.faces, smooth=False, drawEdges = False)
            self.LONI_mesh.translate(0, 1.9, 0)       #改變整個ml的中心位置，ex:(1.3,2.5,-1.2),(1.5,2,0)
            self.LONI_mesh.scale(0.13, 0.1, 0.103)        #改變整個ml在各個方向上的比例，ex:(4.1,4.1,4.1)(3.8,3.8,3.8)
            self.LONI_mesh.opts['color'] = (1.,1.,1.,0.07)
            self.LONI_mesh.setGLOptions('additive')          #透明度啟用
            self.glWidget.addItem(self.LONI_mesh)
        else:
            self.glWidget.removeItem(self.LONI_mesh)
            self.LONI_mesh = gl.GLMeshItem(vertexes=self.verts, faces=self.faces, vertexColors=self.vertex_cols, smooth=False, drawEdges = False)
            self.LONI_mesh.translate(0, 1.9, 0)       #改變整個ml的中心位置，ex:(1.3,2.5,-1.2),(1.5,2,0)
            self.LONI_mesh.scale(0.13, 0.1, 0.103)        #改變整個ml在各個方向上的比例，ex:(4.1,4.1,4.1)(3.8,3.8,3.8)
            self.LONI_mesh.opts['color'] = (1.,1.,1.,0.02)
            self.LONI_mesh.setGLOptions('additive')          #透明度啟用
            self.glWidget.addItem(self.LONI_mesh)
        
    def setup_signal_handler(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.timer_interval*1000)
        self.timer.timeout.connect(self.update_CC_plot)
        self.timer.start()
    
    def update_CC_plot(self):
        if self.ws_CC.update_time_out():
            for line in self.pre_line:
                self.glWidget.removeItem(line)
            self.pre_line = list()
            
            con, event_exist = self.ws_CC.do_connectivity(self.fmin, self.fmax, self.method, self.mode, self.event_id)
            if event_exist == False:
                return
            
            idx = list()
            for i in range(self.ws_CC.channel_num):
                idx.append(i)
            con[idx][:, idx]
            con = con[:,:, 0]
            n_con = self.ws_CC.channel_num
            min_dist = 0.5
            threshold = np.sort(con, axis=None)[-n_con]
            
            ii, jj = np.where(con >= threshold)
            con_nodes = list()
            con_val = list()
            
            for i, j in zip(ii, jj):
                if linalg.norm(self.ch_loc_3D[i] - self.ch_loc_3D[j]) > min_dist:
                    con_nodes.append((i, j))
                    con_val.append(con[i, j])
            con_val = np.array(con_val)
            vmax = round(np.max(con_val), 3)
            vmin = round(np.min(con_val), 3)
            #self.print_system_log("vmax=%.4f         vmin=%.4f" %(vmax, vmin))
            self.norm = mpl.colors.Normalize(vmax=vmax, vmin=vmin)
            self.colorbar = mpl.colorbar.ColorbarBase(self.ax, cmap=self.col_map, norm=self.norm, orientation = 'horizontal', ticks=[vmin, vmax])
            self.colorbar.set_label(self.colorbar_label, labelpad=-43, size=15)
            #self.print_system_log(str(con));
            
            # WS2Unity start send message 開始傳送給unity
            '''
            send information:
                location: nodes[0]、nodes[1]
                value: val 、none(vmax==vmin)
                
            '''
            for val, nodes in zip(con_val, con_nodes):
                x1, y1, z1 = self.ch_loc_3D[nodes[0]]
                x2, y2, z2 = self.ch_loc_3D[nodes[1]]
                connected = np.array([[x1,y1,z1],[x2,y2,z2]])
                if vmax == vmin:
                    line = gl.GLLinePlotItem(pos=connected, width=3.5, color=self.col_map(0.5), antialias=True)
                else:
                    line = gl.GLLinePlotItem(pos=connected, width=3.5, color=self.col_map(self.norm(round(val, 3))),antialias=True)
                self.glWidget.addItem(line)
                self.pre_line.append(line)
            self.canvas.draw()
        
class MyLabelItem(GLGraphicsItem):
    def __init__(self, pos=None, text=None, glWidget=None):
        super().__init__()
        self.text = text
        self.pos = pos
        self.setGLViewWidget(glWidget)

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def paint(self):
        font = QtGui.QFont("Times", 14)
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        for idx in range(0, len(self.text)):
            self.GLViewWidget.renderText(self.pos[idx][0]+0.2, self.pos[idx][1]-0.2, self.pos[idx][2]+0.3, self.text[idx], font)

if __name__ == "__main__":
    app = QtGui.QApplication([])
    plot = CC_Plot()
    app.exec_()