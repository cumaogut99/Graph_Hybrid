"""
Saving and restoring the graph layout (tabs, graphs, signals, settings).

Part of TimeGraphWidget; moved from src/ui/time_graph_widget.py without changes.
"""

import logging
from src.graphics.graph_container import GraphContainer

logger = logging.getLogger(__name__)


class LayoutConfigMixin:
    """Saving and restoring the graph layout (tabs, graphs, signals, settings)."""

    def get_layout_config(self):
        """Mevcut sekme, grafik ve sinyal düzenini bir sözlük olarak alır."""
        tabs_config = []
        
        for i in range(self.tab_widget.count()):
            # Her sekme bir GraphContainer widget'ıdır
            container = self.tab_widget.widget(i)
            
            if not isinstance(container, GraphContainer):
                continue
            
            # Sekme ismini al
            tab_name = self.tab_widget.tabText(i)
                
            # Bu container'daki grafik sayısını al
            subplot_count = container.plot_manager.get_subplot_count()
            
            graphs_config = []
            # Her grafik için sinyal listesini al
            for plot_index in range(subplot_count):
                plot_signals = []
                
                # Bu grafikte görünen sinyalleri bul
                current_signals = container.plot_manager.get_current_signals()
                for signal_key, plot_item in current_signals.items():
                    # signal_key formatı: "signal_name_plot_index"
                    if signal_key.endswith(f"_{plot_index}"):
                        signal_name = '_'.join(signal_key.split('_')[:-1])
                        plot_signals.append(signal_name)
                
                graphs_config.append({'signals': plot_signals})
            
            tabs_config.append({
                'name': tab_name,
                'graphs': graphs_config
            })
            
        return {'tabs': tabs_config}

    def set_layout_config(self, config):
        """Verilen konfigürasyona göre sekme, grafik ve sinyal düzenini ayarlar."""
        
        # Check if config is None (e.g., binary MPAI without layout)
        if config is None:
            logger.info("No layout config provided, keeping default layout")
            return
        
        # Mevcut tüm sekmeleri temizle
        self.tab_widget.clear()
        self.graph_containers.clear()

        if 'tabs' not in config:
            return

        for i, tab_config in enumerate(config['tabs']):
            # Sekme ismini al (varsa kaydedilen ismi, yoksa varsayılan ismi kullan)
            tab_name = tab_config.get('name', f"Tab {i+1}")
            
            # Yeni sekme ekle
            container = self._add_tab(tab_name)
            
            if 'graphs' in tab_config:
                graph_count = len(tab_config['graphs'])
                if graph_count > 0:
                    # Grafik sayısını ayarla
                    container.set_graph_count(graph_count)
                    
                    # UI güncellemesini bekle
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()

                    # Her grafik için sinyalleri ekle
                    for plot_index, graph_config in enumerate(tab_config['graphs']):
                        if 'signals' in graph_config:
                            # Mevcut sinyalleri kontrol et
                            all_signals = set(self.signal_processor.get_signal_names())
                            logger.info(f"[LAYOUT] Tab {i}, Graph {plot_index}: Restoring signals {graph_config['signals']}")
                            logger.info(f"[LAYOUT] Available signals: {len(all_signals)} signals")
                            
                            for signal_name in graph_config['signals']:
                                if signal_name in all_signals:
                                    # Sinyal verilerini signal_processor'dan al
                                    signal_data = self.signal_processor.get_signal_data(signal_name)
                                    
                                    x_data = None
                                    y_data = None
                                    
                                    # Check if data is directly available (CSV mode)
                                    if signal_data and 'x_data' in signal_data and 'y_data' in signal_data:
                                        x_data = signal_data['x_data']
                                        y_data = signal_data['y_data']
                                    # Otherwise, try to get downsampled view (MPAI mode)
                                    elif signal_data and 'mpai_reader' in signal_data:
                                        logger.debug(f"[LAYOUT] Getting downsampled view for memory-mapped signal '{signal_name}'")
                                        ds_view = self.signal_processor.get_downsampled_view(signal_name)
                                        if ds_view and 'x_data' in ds_view and 'y_data' in ds_view:
                                            x_data = ds_view['x_data']
                                            y_data = ds_view['y_data']
                                    
                                    if x_data is not None and y_data is not None:
                                        # Sinyali belirtilen grafiğe ekle
                                        container.add_signal(signal_name, x_data, y_data, plot_index)
                                        logger.info(f"[LAYOUT] Added signal '{signal_name}' to graph {plot_index}")
                                        
                                        # Signal mapping'i güncelle
                                        if i not in self.graph_signal_mapping:
                                            self.graph_signal_mapping[i] = {}
                                        if plot_index not in self.graph_signal_mapping[i]:
                                            self.graph_signal_mapping[i][plot_index] = []
                                        if signal_name not in self.graph_signal_mapping[i][plot_index]:
                                            self.graph_signal_mapping[i][plot_index].append(signal_name)
                                    else:
                                        logger.warning(f"[LAYOUT] Could not get data for signal '{signal_name}'")
                                else:
                                    logger.warning(f"[LAYOUT] Signal '{signal_name}' not found in signal_processor")
                    
                    # Bu sekme için statistics panel'i güncelle
                    if i == self.tab_widget.currentIndex():  # Sadece aktif sekme için
                        self._recreate_statistics_panel()
                        # DON'T update statistics here - wait for cursor movement
