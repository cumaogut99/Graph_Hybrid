#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilimsel Veri Dosyası İçe Aktarıcı (TDM / TDX / TDMS)
=====================================================

Bu modül, National Instruments tarafından kullanılan TDMS ve TDM/TDX
dosyalarını uygulamanın geri kalanının anlayabileceği bir ara CSV
dosyasına dönüştürür.

Mimari Karar:
-------------
CSV dosyaları için zaten olgunlaşmış bir boru hattı (pipeline) mevcut:

    Dosya Seç -> DataImportDialog (önizleme + zaman/header/satır seçimi)
              -> CsvToMpaiConverter (MPAI üretimi + ilerleme overlay'i)
              -> MPAI üzerinden görüntüleme

TDM/TDX/TDMS dosyalarını **önce bir ara CSV'ye** dönüştürüp bu boru hattını
yeniden kullanıyoruz. Böylece kullanıcı:
  * Aynı pencerede (DataImportDialog) önizleme görür,
  * CSV'de olduğu gibi zaman kolonu / header satırı / ilk veri satırı seçer,
  * Veri yine MPAI formatına dönüştürülür ve MPAI üzerinden işlenir.

Dönüştürme sırasında ilerleme yüzdesi `progress` sinyali ile bir overlay'e
aktarılır.

Notlar:
  * TDMS okuma `npTDMS` kütüphanesi ile yapılır (sağlam, yaygın kullanılır).
  * TDM/TDX okuma, NI USI 1.0 şemasına göre yazılmış bir XML + ikili (binary)
    ayrıştırıcı ile yapılır. Standart sayısal TDM/TDX dosyaları desteklenir.
"""

import os
import csv
import logging
import xml.etree.ElementTree as ET

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal as Signal

logger = logging.getLogger(__name__)

# İçe aktarmayı desteklediğimiz bilimsel dosya uzantıları
SCIENTIFIC_EXTENSIONS = {'.tdms', '.tdm', '.tdx'}

# Ara CSV yazılırken kullanılan satır yığını boyutu (bellek dostu akış)
_CSV_CHUNK_ROWS = 100_000


def is_scientific_file(file_path: str) -> bool:
    """Verilen yolun desteklenen bir TDM/TDX/TDMS dosyası olup olmadığını döndürür."""
    if not file_path:
        return False
    return os.path.splitext(file_path)[1].lower() in SCIENTIFIC_EXTENSIONS


# NI USI 1.0 valueType -> numpy dtype karakteri eşlemesi
_USI_VALUE_TYPES = {
    'eInt8Usi': 'i1', 'eInt16Usi': 'i2', 'eInt32Usi': 'i4', 'eInt64Usi': 'i8',
    'eUInt8Usi': 'u1', 'eUInt16Usi': 'u2', 'eUInt32Usi': 'u4', 'eUInt64Usi': 'u8',
    'eFloat32Usi': 'f4', 'eFloat64Usi': 'f8',
}


class ScientificImportError(Exception):
    """Bilimsel dosya dönüştürme sırasında oluşan hatalar."""


class ScientificFileConverter(QObject):
    """
    Bir TDM/TDX/TDMS dosyasını ara bir CSV dosyasına dönüştüren işçi (worker).

    Sinyaller:
        progress(str, int) : (mesaj, yüzde 0-100)
        finished(str)      : oluşturulan CSV yolu
        error(str)         : hata mesajı
    """

    progress = Signal(str, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_path: str, output_csv_path: str):
        super().__init__()
        self.input_path = input_path
        self.output_csv_path = output_csv_path
        self.cancelled = False
        # DataOperations._on_cancel_loading bu özniteliğe erişebildiği için
        # güvenli bir varsayılan sağlıyoruz.
        self.settings = {}

    def cancel(self):
        """Dönüştürmeyi iptal et."""
        self.cancelled = True
        logger.info("Scientific file conversion cancelled by user")

    # ------------------------------------------------------------------ #
    # Qt thread giriş noktaları
    # ------------------------------------------------------------------ #
    def run(self):
        """QThread.started ile bağlanan giriş noktası."""
        self.convert()

    def convert(self) -> bool:
        """Dosyayı ara CSV'ye dönüştür. Başarılı olursa True döndürür."""
        try:
            if not os.path.exists(self.input_path):
                raise ScientificImportError(f"Dosya bulunamadı: {self.input_path}")

            ext = os.path.splitext(self.input_path)[1].lower()
            self.progress.emit(f"{os.path.basename(self.input_path)} okunuyor...", 1)

            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)

            if ext == '.tdms':
                self._convert_tdms()
            elif ext in ('.tdm', '.tdx'):
                self._convert_tdm_tdx()
            else:
                raise ScientificImportError(f"Desteklenmeyen format: {ext}")

            if self.cancelled:
                self._cleanup_partial_output()
                # İptal de bir "bitiş"tir: thread'in temizlenebilmesi için sinyal ver.
                self.error.emit("İçe aktarma kullanıcı tarafından iptal edildi")
                return False

            self.progress.emit("Dönüştürme tamamlandı", 100)
            self.finished.emit(self.output_csv_path)
            return True

        except Exception as e:  # noqa: BLE001 - kullanıcıya anlamlı mesaj göster
            logger.exception("Scientific file conversion failed")
            self._cleanup_partial_output()
            self.error.emit(str(e))
            return False

    def _cleanup_partial_output(self):
        try:
            if os.path.exists(self.output_csv_path):
                os.remove(self.output_csv_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # TDMS (npTDMS)
    # ------------------------------------------------------------------ #
    def _convert_tdms(self):
        """TDMS dosyasını npTDMS ile okuyup ara CSV'ye yazar."""
        try:
            from nptdms import TdmsFile
        except ImportError:
            raise ScientificImportError(
                "TDMS desteği için 'npTDMS' kütüphanesi gerekli. "
                "Lütfen kurun: pip install nptdms"
            )

        self.progress.emit("TDMS yapısı çözümleniyor...", 5)

        # Akış (streaming) modu: tüm dosya belleğe yüklenmez.
        with TdmsFile.open(self.input_path) as tdms:
            columns = self._collect_tdms_columns(tdms)
            if not columns:
                raise ScientificImportError("TDMS dosyasında okunabilir kanal bulunamadı.")

            max_len = max((c['length'] for c in columns), default=0)
            if max_len == 0:
                raise ScientificImportError("TDMS kanallarında veri (örnek) bulunamadı.")

            # Dalga formu (waveform) zamanlaması varsa açık bir "Time" kolonu üret.
            time_info = self._tdms_time_info(columns)

            header = ([time_info['name']] if time_info else []) + [c['name'] for c in columns]

            self.progress.emit("TDMS verisi CSV'ye yazılıyor...", 10)
            self._stream_write_csv(
                header=header,
                total_rows=max_len,
                chunk_reader=lambda start, n: self._read_tdms_chunk(columns, time_info, start, n),
                progress_lo=10,
                progress_hi=99,
            )

    def _collect_tdms_columns(self, tdms):
        """Kanalları toplar ve kolon adlarındaki çakışmaları çözer."""
        raw = []
        name_counts = {}
        for group in tdms.groups():
            for ch in group.channels():
                name_counts[ch.name] = name_counts.get(ch.name, 0) + 1

        for group in tdms.groups():
            for ch in group.channels():
                try:
                    length = len(ch)
                except (TypeError, KeyError):
                    length = ch.properties.get('NI_ArrayColumn_Length', 0) or 0
                # Aynı isim birden çok grupta varsa grup adıyla nitele.
                if name_counts.get(ch.name, 0) > 1:
                    col_name = f"{group.name}/{ch.name}"
                else:
                    col_name = ch.name
                raw.append({
                    'name': self._clean_name(col_name),
                    'channel': ch,
                    'length': int(length),
                    'properties': dict(ch.properties),
                })
        return raw

    @staticmethod
    def _clean_name(name) -> str:
        cleaned = str(name).strip().replace('\n', ' ').replace('\r', ' ')
        # CSV başlıklarında virgül sorun çıkarmasın diye boşlukla değiştir.
        cleaned = cleaned.replace(',', ' ')
        return cleaned or 'channel'

    def _tdms_time_info(self, columns):
        """
        Dalga formu özelliklerinden (wf_increment / wf_start_offset) bir zaman
        ekseni üretilebiliyorsa bilgiyi döndürür, aksi halde None.
        """
        for c in columns:
            props = c['properties']
            inc = props.get('wf_increment')
            if inc:
                try:
                    inc = float(inc)
                except (TypeError, ValueError):
                    continue
                if inc > 0:
                    start = props.get('wf_start_offset', 0.0)
                    try:
                        start = float(start)
                    except (TypeError, ValueError):
                        start = 0.0
                    return {'name': 'Time', 'increment': inc, 'start': start}
        return None

    def _read_tdms_chunk(self, columns, time_info, start, n):
        """[start, start+n) aralığı için kolon dizilerini döndürür."""
        arrays = []
        if time_info:
            idx = np.arange(start, start + n, dtype=np.float64)
            arrays.append(time_info['start'] + idx * time_info['increment'])

        for c in columns:
            length = c['length']
            if start >= length:
                arrays.append(np.full(n, np.nan, dtype=np.float64))
                continue
            count = min(n, length - start)
            try:
                data = c['channel'].read_data(start, count, scaled=True)
            except TypeError:
                # Eski npTDMS sürümleri 'scaled' parametresini desteklemeyebilir.
                data = c['channel'].read_data(start, count)
            data = np.asarray(data)
            arrays.append(self._normalize_chunk(data, n))
        return arrays

    @staticmethod
    def _normalize_chunk(data, n):
        """Bir kanal yığınını n uzunluğunda, yazılabilir bir diziye normalize eder."""
        if data.dtype.kind in 'fiu':
            out = np.full(n, np.nan, dtype=np.float64)
            out[:len(data)] = data.astype(np.float64, copy=False)
            return out
        # Sayısal olmayan (string, datetime vb.) veriler için metin temsili.
        out = np.empty(n, dtype=object)
        out[:] = ''
        for i, v in enumerate(data[:n]):
            out[i] = v
        return out

    # ------------------------------------------------------------------ #
    # TDM / TDX (NI USI 1.0)
    # ------------------------------------------------------------------ #
    def _convert_tdm_tdx(self):
        """TDM (XML başlık) + TDX (ikili veri) çiftini ara CSV'ye dönüştürür."""
        tdm_path, tdx_path = self._resolve_tdm_pair(self.input_path)
        self.progress.emit("TDM başlığı çözümleniyor...", 5)

        columns, byte_order = self._parse_tdm_header(tdm_path, tdx_path)
        if not columns:
            raise ScientificImportError(
                "TDM dosyasında okunabilir sayısal kanal bulunamadı. "
                "(Yalnızca standart NI USI 1.0 TDM/TDX dosyaları desteklenir.)"
            )

        max_len = max((c['length'] for c in columns), default=0)
        if max_len == 0:
            raise ScientificImportError("TDM kanallarında veri (örnek) bulunamadı.")

        header = [c['name'] for c in columns]
        self.progress.emit("TDX verisi CSV'ye yazılıyor...", 10)
        self._stream_write_csv(
            header=header,
            total_rows=max_len,
            chunk_reader=lambda start, n: self._read_tdm_chunk(columns, start, n),
            progress_lo=10,
            progress_hi=99,
        )

    @staticmethod
    def _resolve_tdm_pair(path):
        """Verilen .tdm veya .tdx yolundan (tdm, tdx) çiftini bulur."""
        stem, ext = os.path.splitext(path)
        if ext.lower() == '.tdm':
            tdm_path = path
            tdx_path = stem + '.tdx'
        else:  # .tdx
            tdx_path = path
            tdm_path = stem + '.tdm'
        if not os.path.exists(tdm_path):
            raise ScientificImportError(
                f"TDM başlık dosyası bulunamadı: {os.path.basename(tdm_path)}\n"
                "TDM ve TDX dosyaları aynı klasörde, aynı isimde olmalıdır."
            )
        if not os.path.exists(tdx_path):
            raise ScientificImportError(
                f"TDX veri dosyası bulunamadı: {os.path.basename(tdx_path)}\n"
                "TDM ve TDX dosyaları aynı klasörde, aynı isimde olmalıdır."
            )
        return tdm_path, tdx_path

    @staticmethod
    def _local_tag(elem) -> str:
        """XML etiketini namespace olmadan döndürür."""
        tag = elem.tag
        return tag.split('}', 1)[1] if '}' in tag else tag

    def _parse_tdm_header(self, tdm_path, tdx_path):
        """
        TDM XML başlığını çözümleyip kanal listesini oluşturur.

        Her kanal:
            {name, dtype(np), byte_offset, length, block_offset}
        Zincir:  tdm_channel -> localcolumn -> *_sequence -> block(external)
        """
        tree = ET.parse(tdm_path)
        root = tree.getroot()

        elements = list(root.iter())
        by_id = {}
        for el in elements:
            el_id = el.get('id')
            if el_id:
                by_id[el_id] = el

        # 1) byte sırası (byteOrder) ve blok tablosu
        byte_order = '<'  # littleEndian varsayılan
        blocks = {}
        for el in elements:
            tag = self._local_tag(el)
            if tag == 'file' and el.get('byteOrder'):
                byte_order = '>' if 'big' in el.get('byteOrder', '').lower() else '<'
            if tag in ('block', 'block_bench') and el.get('byteOffset') is not None:
                blk_id = el.get('id')
                if blk_id:
                    blocks[blk_id] = {
                        'byte_offset': int(el.get('byteOffset', 0)),
                        'length': int(el.get('length', 0)),
                        'value_type': el.get('valueType', 'eFloat64Usi'),
                        'block_offset': int(el.get('blockOffset', 0) or 0),
                    }

        # 2) sequence id -> block id  (values external="...")
        seq_to_block = {}
        for el in elements:
            tag = self._local_tag(el)
            if tag.endswith('_sequence') and el.get('id'):
                ext = None
                for child in el.iter():
                    if self._local_tag(child) == 'values' and child.get('external'):
                        ext = child.get('external')
                        break
                if ext:
                    seq_to_block[el.get('id')] = ext

        # 3) localcolumn id -> sequence id
        lc_to_seq = {}
        for el in elements:
            if self._local_tag(el) != 'localcolumn' or not el.get('id'):
                continue
            seq_ref = self._find_ref(el, 'values', by_id)
            if seq_ref:
                lc_to_seq[el.get('id')] = seq_ref

        # 4) kanalları sırayla işle
        columns = []
        used = set()
        for el in elements:
            if self._local_tag(el) != 'tdm_channel':
                continue
            name = self._channel_name(el)
            lc_ref = self._find_ref(el, 'local_columns', by_id) or \
                self._find_ref(el, 'measurement_quantity', by_id)
            # measurement_quantity -> localcolumn dolaylı olabilir; doğrudan da deneriz.
            seq_id = lc_to_seq.get(lc_ref)
            if seq_id is None and lc_ref in seq_to_block:
                seq_id = lc_ref
            block_id = seq_to_block.get(seq_id)
            block = blocks.get(block_id)
            if not block:
                logger.warning("TDM channel '%s' has no resolvable data block, skipping", name)
                continue
            dtype_char = _USI_VALUE_TYPES.get(block['value_type'])
            if dtype_char is None:
                logger.warning("TDM channel '%s' has unsupported valueType '%s', skipping",
                               name, block['value_type'])
                continue
            col_name = self._clean_name(name)
            base = col_name
            k = 1
            while col_name in used:
                col_name = f"{base}_{k}"
                k += 1
            used.add(col_name)
            columns.append({
                'name': col_name,
                'dtype': np.dtype(byte_order + dtype_char),
                'byte_offset': block['byte_offset'],
                'length': block['length'],
                'tdx_path': tdx_path,
            })
        return columns, byte_order

    def _channel_name(self, channel_el):
        for child in channel_el.iter():
            if self._local_tag(child) == 'name' and (child.text or '').strip():
                return child.text.strip()
        return channel_el.get('id', 'channel')

    def _find_ref(self, parent_el, child_tag, by_id):
        """
        Bir alt etiketin referans verdiği id'yi döndürür. Referans hem etiket
        metni (id) hem de 'idref'/'usi:ref' özniteliği olarak gelebilir.
        """
        for child in parent_el.iter():
            if self._local_tag(child) != child_tag:
                continue
            for attr in ('idref', 'ref', '{http://www.ni.com/Schemas/USI/1_0}ref'):
                if child.get(attr):
                    return child.get(attr).split()[0]
            if (child.text or '').strip():
                return child.text.strip().split()[0]
        return None

    def _read_tdm_chunk(self, columns, start, n):
        """TDX dosyasından [start, start+n) aralığını okur."""
        arrays = []
        for c in columns:
            length = c['length']
            if start >= length:
                arrays.append(np.full(n, np.nan, dtype=np.float64))
                continue
            count = min(n, length - start)
            itemsize = c['dtype'].itemsize
            offset = c['byte_offset'] + start * itemsize
            with open(c['tdx_path'], 'rb') as f:
                f.seek(offset)
                data = np.fromfile(f, dtype=c['dtype'], count=count)
            arrays.append(self._normalize_chunk(data, n))
        return arrays

    # ------------------------------------------------------------------ #
    # Ortak CSV akış yazıcısı
    # ------------------------------------------------------------------ #
    def _stream_write_csv(self, header, total_rows, chunk_reader, progress_lo, progress_hi):
        """
        Veriyi satır yığınları halinde ara CSV'ye yazar ve ilerleme bildirir.

        Args:
            header: kolon adları listesi (zaman kolonu dahil)
            total_rows: toplam satır sayısı
            chunk_reader: (start, n) -> kolon dizileri listesi (header sırasında)
            progress_lo/hi: bu aşamanın yüzde aralığı
        """
        rows_written = 0
        with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(header)

            start = 0
            while start < total_rows:
                if self.cancelled:
                    logger.info("CSV streaming cancelled at row %d", start)
                    return
                n = min(_CSV_CHUNK_ROWS, total_rows - start)
                arrays = chunk_reader(start, n)

                # Sütun dizilerini satır satır yaz.
                cols = [self._format_column(a) for a in arrays]
                for i in range(n):
                    writer.writerow([col[i] for col in cols])

                rows_written += n
                start += n
                pct = progress_lo + int((rows_written / max(total_rows, 1)) * (progress_hi - progress_lo))
                self.progress.emit(
                    f"CSV'ye yazılıyor... ({rows_written:,}/{total_rows:,} satır)",
                    min(pct, progress_hi),
                )
        logger.info("Intermediate CSV written: %s (%d rows)", self.output_csv_path, rows_written)

    @staticmethod
    def _format_column(arr):
        """Bir kolon dizisini CSV'ye yazılabilir string listesine çevirir."""
        if arr.dtype.kind == 'f':
            # NaN -> boş hücre; aksi halde kompakt sayı temsili.
            out = []
            for v in arr:
                out.append('' if np.isnan(v) else repr(float(v)))
            return out
        return ['' if v is None else str(v) for v in arr]
