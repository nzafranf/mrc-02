# Error sample: sliding_window (seed13)

40 errors stratified by outcome (round robin) + 10 correct answers with the highest na_prob. Fill human_label with one of: gold_issue, multi_hop_miss, right_type_wrong_entity, wrong_type, none, other.

| # | id | outcome | question | gold | pred | q_type | human_label | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | qa_a34f8d360ca43b5da17a | wrong_sentence | Kapan Hayao Miyazaki mulai menjadi sutradara ? | 1971 | 1979 | time |  |  |
| 2 | qa_1b3c69b46bf0a761fa40 | false_answer_on_unanswerable | Berapa jumlah provinsi di pulau Selatan? | (no answer) | Jawa Barat | number |  |  |
| 3 | qa_29051b5744cbc451d3ae | wrong_span_same_sentence | Apa nama museum terbesar di Indonesia ? | Museum Nasional Republik Indonesia | Jakarta Pusat | entity_or_def |  |  |
| 4 | qa_56f272b5758492c87f78 | boundary_too_short | siapakah karakter utama dalam manga One Piece? | Monkey D. Luffy | Monkey D | person |  |  |
| 5 | qa_d2a6e12f209bf3a30b23 | partial_overlap | Apakah maksud istilah Shengnü? | wanita sisa / ejekan yang dipopulerkan oleh Federasi Wanita Seluruh Tiongkok yang mengklasifikasikan wanita yang masih belum menikah pada usia akhir dua puluh tahunan atau lebih | Federasi Wanita Seluruh Tiongkok | entity_or_def |  |  |
| 6 | qa_3c0aad94510547076e78 | false_no_answer | Siapa pendiri perusahaan Yahoo!? | Jerry Yang dan David Filo | (no answer) | person |  |  |
| 7 | qa_8f074654d605ff33022f | boundary_too_long | Kapan dinasti Chandragupta berdiri ? | 322 SM | Pada 322 SM | time |  |  |
| 8 | qa_20ef37ed8d030f583cee | wrong_sentence | berapakah luas Kanada? | 9.970.610 kilometer persegi | Amerika Utara | number |  |  |
| 9 | qa_2ab804f8b8e0c71c1d42 | false_answer_on_unanswerable | Kapan perdagangan melalui armada mulai tidak dilakukan oleh bangsa Eropa? | (no answer) | Portugal | time |  |  |
| 10 | qa_b447b32d8614b5af9993 | wrong_span_same_sentence | Apa maksud kata Wangsa? | kelanjutan kekuasaan pemerintahan yang dipegang oleh satu garis keturunan | dinasti | entity_or_def |  |  |
| 11 | qa_48ebb5112d59ca9bbe6f | boundary_too_short | Mengapa emas batangan digunakan untuk tujuan investasi? | Harga emas di prediksi akan terus mengalami kenaikan karena tidak adanya kepastian kapan berakhirnya krisis hutang yang melilit negara-negara eropa, dan penurunan tingkat ekonomi dari 9 negara anggota uni-Eropa, sehingga harga saham di berbagai bursa di dunia mengalami kejatuhan | 9 | other |  |  |
| 12 | qa_0d82790c6d3244970680 | partial_overlap | siapakah orang tua Mahasthavira Vajragiri? | Ayahnya bernama The Tjai Gwan dan ibunya Liem Kiat Nio | The Tjing San | person |  |  |
| 13 | qa_a62b7545d71fb68f6528 | false_no_answer | Kapan Yahoo! Inc. berdiri ? | Januari 1994 | (no answer) | time |  |  |
| 14 | qa_fdbd93a2c835fe1e3efd | boundary_too_long | Siapa yang memenangkan Pertempuran Tanjung Esperance? | Sekutu | Akhirnya Sekutu | person |  |  |
| 15 | qa_94940c9fa1996638c38b | wrong_sentence | Apa sistem pemerintahan yang digunakan di Libanon? | republik demokratis parlementer | keagamaan | entity_or_def |  |  |
| 16 | qa_adb1cd99f76ad21e9443 | false_answer_on_unanswerable | Kapan KeshoGUN Taisel didirikan? | (no answer) | Jepang | time |  |  |
| 17 | qa_1736e82de25b5f57abdc | wrong_span_same_sentence | Pada suhu berapa air mendidih ? | 0°C | 0 | number |  |  |
| 18 | qa_175e4e7028a6638527bf | boundary_too_short | dimanakah letak gunung Fuji di Jepang ? | perbatasan Prefektur Shizuoka dan Yamanashi, di sebelah barat Tokyo | Prefektur Shizuoka | location |  |  |
| 19 | qa_20564c45de2060da8fb1 | partial_overlap | Kapan Yesus lahir ? | 6 SM | 1 SM | time |  |  |
| 20 | qa_72aa719a1d4361fc2454 | false_no_answer | Siapakah pendiri Yahoo.inc? | Jerry Yang dan David Filo | (no answer) | person |  |  |
| 21 | qa_88ebde639e610f7116e7 | boundary_too_long | Berapa banyak daerah Indonesia yang tidak dihuni dan ditutupi hutan tropis? | 45% | Walaupun 45 | number |  |  |
| 22 | qa_51c7159a8e72a21f29c6 | wrong_sentence | Dimana  Sultan Rakyatullah lahir ? | Tambangan | Raden Halit | location |  |  |
| 23 | qa_3ddf60850d6573e1050d | false_answer_on_unanswerable | Apakah efek samping vitamin B? | (no answer) | 21 | entity_or_def |  |  |
| 24 | qa_78b44ef45d2e38e26d29 | wrong_span_same_sentence | berapakah jumlah populasi di seoul? | 10 juta | 605.21 | number |  |  |
| 25 | qa_3613ed18aa06c112d456 | boundary_too_short | Apa definisi teologi? | Teologi (bahasa Yunani θεος, theos, "], Tuhan", dan λογια, logia, "kata-kata," "ucapan," atau "wacana") adalah wacana yang berdasarkan nalar mengenai agama, spiritualitas dan Tuhan (Lih. bawah, "Teologi dan agama-agama lain di luar agama Kristen"). | Yunani | entity_or_def |  |  |
| 26 | qa_981abd2dceccf3213d45 | partial_overlap | Apakah yang dimaksud dengan alamat IP? | deretan angka biner antara 32 bit sampai 128 bit yang dipakai sebagai alamat identifikasi untuk tiap komputer host dalam jaringan Internet | Internet Protocol Address | entity_or_def |  |  |
| 27 | qa_3756dd03ea79c52b0a61 | false_no_answer | Siapa nama pendiri Yahoo! Inc.? | Jerry Yang dan David Filo | (no answer) | person |  |  |
| 28 | qa_d509e14f88c949d14a59 | boundary_too_long | dimanakah internet pertama kali diciptakan? | Amerika Serikat | Departemen Pertahanan Amerika Serikat | location |  |  |
| 29 | qa_5f9a7f08aa0f1cd7fe77 | wrong_sentence | apakah nama Pesawat yang terbesar pernah dibuat? | Airbus A380 | Superjumbo | entity_or_def |  |  |
| 30 | qa_0c11962e96fb79592a56 | false_answer_on_unanswerable | Berapakah jumlah populasi anak muda di Seoul mengacu sensus 2014? | (no answer) | 605.21 | number |  |  |
| 31 | qa_343645a3c70b99a9a379 | wrong_span_same_sentence | kapan kah Bahasa Sanskerta masuk ke idnonesia? | ribuan tahun lalu | Indonesia | time |  |  |
| 32 | qa_cca3bea09e75907fa206 | boundary_too_short | Kapan Pertempuran Tanjung Esperance terjadi ? | 11 Oktober hingga 12 Oktober 1942 | 11 Oktober | time |  |  |
| 33 | qa_09df0b3461ac400e9981 | partial_overlap | kapankah Yesus Kristus dilahirkan? | sesudah tahun 6 SM | 1 SM | time |  |  |
| 34 | qa_4b1b34abe018e7c8a821 | false_no_answer | Kapan perusahaan Yahoo! berdiri ? | Januari 1994 | (no answer) | time |  |  |
| 35 | qa_cd1b03ee6bba5cd06e5f | boundary_too_long | dimanakah letak YPBB? | Bandung | Yayasan Pendidikan Bale Bandung | location |  |  |
| 36 | qa_39d119dfef45812d53ba | wrong_sentence | Apa yang dimaksud dengan neokonservatif ? | aliran politik yang lahir di Amerika Serikat pada tahun 1960-an di kalangan Demokrat yang tidak sepakat dengan kebijakan dalam dan luar negeri partainya | George W | entity_or_def |  |  |
| 37 | qa_126379bbfd9c2621c543 | false_answer_on_unanswerable | Kapan Republik Gambia dijajah? | (no answer) | 18 Februari 1965 | time |  |  |
| 38 | qa_6397b901d4f6f33fd703 | wrong_span_same_sentence | Apa fungsi Peptida antimikrobial? | immunomodulator | konvensional | entity_or_def |  |  |
| 39 | qa_35ee441f328f27f59fd2 | boundary_too_short | siapakah karakter utama dalam One Piece? | Monkey D. Luffy | Monkey D | person |  |  |
| 40 | qa_15fcf4146498cca19dcf | partial_overlap | Siapakah yang tidak sedikit memberikan kontribusi bagi berdirinya Dinasti Tang? | Li Jiancheng | Li Shimin | person |  |  |
| 41 | qa_006703b670a2de86a02f | correct_low_conf | kapankah Uni Eropa didirikan? | 1993 | 1993 | time |  |  |
| 42 | qa_01364a9092f96eb6e2be | correct_low_conf | Apakah kota terbesar di Iran? | Teheran | Teheran | entity_or_def |  |  |
| 43 | qa_05aed54e82bdb9edebf5 | correct_low_conf | Apakah ibukota Papua Nugini? | Port Moresby | Port Moresby | entity_or_def |  |  |
| 44 | qa_07b2bea085cd5a148a56 | correct_low_conf | Sejak tanggal berapakah Jepang tidak berhak mencampuri urusan perkeretaapian di Indonesia? | 28 September 1945 | 28 September 1945 | number |  |  |
| 45 | qa_117ffbb8cd4ed821d9e9 | correct_low_conf | apakah nama ibukota Vietnam? | Hanoi | Hanoi | entity_or_def |  |  |
| 46 | qa_11f43b983ed2d07230d3 | correct_low_conf | apakah nama ibukota Israel? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
| 47 | qa_120d3e2aeafdd83b1379 | correct_low_conf | Kapan Minecraft pertama kali diciptakan? | 17 Mei 2011 | 17 Mei 2011 | time |  |  |
| 48 | qa_12a96c12a171ddb1849c | correct_low_conf | Apa yang tidak ada di kereta ekonomi sebelum tahun 2010? | AC | AC | entity_or_def |  |  |
| 49 | qa_1a05439cf9633a04e7eb | correct_low_conf | apakah nama ibukota Polandia ? | Warsawa | Warsawa | entity_or_def |  |  |
| 50 | qa_1e0375bb9d9e7cb9488b | correct_low_conf | Apa ibukota Israel ? | Tel Aviv | Tel Aviv | entity_or_def |  |  |
