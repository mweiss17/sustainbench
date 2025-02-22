from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sustainbench.datasets.sustainbench_dataset import SustainBenchDataset
from sustainbench.common.metrics.all_metrics import MSE, PearsonCorrelation
from sustainbench.common.grouper import CombinatorialGrouper
from sustainbench.common.utils import subsample_idxs, shuffle_arr

DATASET = '2009-17'
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']

SPLITS = {
    'train': [
        'AL', 'BD', 'CD', 'CM', 'GH', 'GU', 'HN', 'IA', 'ID', 'JO', 'KE', 'KM',
        'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 'NG', 'NI', 'PE', 'PH',
        'SN', 'TG', 'TJ', 'UG', 'ZM', 'ZW'],
    'val': [
        'BF', 'BJ', 'BO', 'CO', 'DR', 'GA', 'GN', 'GY', 'HT', 'NM', 'SL', 'TD',
        'TZ'],
    'test': [
        'AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'KY', 'ML', 'NP', 'PK', 'RW',
        'SZ']
}

DHS_COUNTRIES = [
    'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
    'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
    'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
    'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']

DHS_COUNTRY_MAP = {'afghanistan': 'AF', 'åland_islands': 'AX', 'albania': 'AL', 'algeria': 'DZ', 'american_samoa': 'AS', 'andorra': 'AD', 'angola': 'AO', 'anguilla': 'AI', 'antarctica': 'AQ', 'antigua_and_barbuda': 'AG', 'argentina': 'AR', 'armenia': 'AM', 'aruba': 'AW', 'australia': 'AU', 'austria': 'AT', 'azerbaijan': 'AZ', 'bahamas': 'BS', 'bahrain': 'BH', 'bangladesh': 'BD', 'barbados': 'BB', 'belarus': 'BY', 'belgium': 'BE', 'belize': 'BZ', 'benin': 'BJ', 'bermuda': 'BM', 'bhutan': 'BT', 'bolivia,_plurinational_state_of': 'BO', 'bonaire,_sint_eustatius_and_saba': 'BQ', 'bosnia_and_herzegovina': 'BA', 'botswana': 'BW', 'bouvet_island': 'BV', 'brazil': 'BR', 'british_indian_ocean_territory': 'IO', 'brunei_darussalam': 'BN', 'bulgaria': 'BG', 'burkina_faso': 'BF', 'burundi': 'BI', 'cambodia': 'KH', 'cameroon': 'CM', 'canada': 'CA', 'cape_verde': 'CV', 'cayman_islands': 'KY', 'central_african_republic': 'CF', 'chad': 'TD', 'chile': 'CL', 'china': 'CN', 'christmas_island': 'CX', 'cocos_(keeling)_islands': 'CC', 'colombia': 'CO', 'comoros': 'KM', 'congo': 'CG', 'democratic_republic_of_congo': 'CD', 'cook_islands': 'CK', 'costa_rica': 'CR', "cote_d_ivoire": 'CI', 'croatia': 'HR', 'cuba': 'CU', 'curaçao': 'CW', 'cyprus': 'CY', 'czech_republic': 'CZ', 'denmark': 'DK', 'djibouti': 'DJ', 'dominica': 'DM', 'dominican_republic': 'DO', 'ecuador': 'EC', 'egypt': 'EG', 'el_salvador': 'SV', 'equatorial_guinea': 'GQ', 'eritrea': 'ER', 'estonia': 'EE', 'ethiopia': 'ET', 'falkland_islands_(malvinas)': 'FK', 'faroe_islands': 'FO', 'fiji': 'FJ', 'finland': 'FI', 'france': 'FR', 'french_guiana': 'GF', 'french_polynesia': 'PF', 'french_southern_territories': 'TF', 'gabon': 'GA', 'gambia': 'GM', 'georgia': 'GE', 'germany': 'DE', 'ghana': 'GH', 'gibraltar': 'GI', 'greece': 'GR', 'greenland': 'GL', 'grenada': 'GD', 'guadeloupe': 'GP', 'guam': 'GU', 'guatemala': 'GT', 'guernsey': 'GG', 'guinea': 'GN', 'guinea-bissau': 'GW', 'guyana': 'GY', 'haiti': 'HT', 'heard_island_and_mcdonald_islands': 'HM', 'holy_see_(vatican_city_state)': 'VA', 'honduras': 'HN', 'hong_kong': 'HK', 'hungary': 'HU', 'iceland': 'IS', 'india': 'IN', 'indonesia': 'ID', 'iran,_islamic_republic_of': 'IR', 'iraq': 'IQ', 'ireland': 'IE', 'isle_of_man': 'IM', 'israel': 'IL', 'italy': 'IT', 'jamaica': 'JM', 'japan': 'JP', 'jersey': 'JE', 'jordan': 'JO', 'kazakhstan': 'KZ', 'kenya': 'KE', 'kiribati': 'KI', "korea,_democratic_people's_republic_of": 'KP', 'korea,_republic_of': 'KR', 'kuwait': 'KW', 'kyrgyzstan': 'KG', "lao_people's_democratic_republic": 'LA', 'latvia': 'LV', 'lebanon': 'LB', 'lesotho': 'LS', 'liberia': 'LR', 'libya': 'LY', 'liechtenstein': 'LI', 'lithuania': 'LT', 'luxembourg': 'LU', 'macao': 'MO', 'macedonia,_the_former_yugoslav_republic_of': 'MK', 'madagascar': 'MG', 'malawi': 'MW', 'malaysia': 'MY', 'maldives': 'MV', 'mali': 'ML', 'malta': 'MT', 'marshall_islands': 'MH', 'martinique': 'MQ', 'mauritania': 'MR', 'mauritius': 'MU', 'mayotte': 'YT', 'mexico': 'MX', 'micronesia,_federated_states_of': 'FM', 'moldova,_republic_of': 'MD', 'monaco': 'MC', 'mongolia': 'MN', 'montenegro': 'ME', 'montserrat': 'MS', 'morocco': 'MA', 'mozambique': 'MZ', 'myanmar': 'MM', 'namibia': 'NA', 'nauru': 'NR', 'nepal': 'NP', 'netherlands': 'NL', 'new_caledonia': 'NC', 'new_zealand': 'NZ', 'nicaragua': 'NI', 'niger': 'NE', 'nigeria': 'NG', 'niue': 'NU', 'norfolk_island': 'NF', 'northern_mariana_islands': 'MP', 'norway': 'NO', 'oman': 'OM', 'pakistan': 'PK', 'palau': 'PW', 'palestine,_state_of': 'PS', 'panama': 'PA', 'papua_new_guinea': 'PG', 'paraguay': 'PY', 'peru': 'PE', 'philippines': 'PH', 'pitcairn': 'PN', 'poland': 'PL', 'portugal': 'PT', 'puerto_rico': 'PR', 'qatar': 'QA', 'réunion': 'RE', 'romania': 'RO', 'russian_federation': 'RU', 'rwanda': 'RW', 'saint_barthélemy': 'BL', 'saint_helena,_ascension_and_tristan_da_cunha': 'SH', 'saint_kitts_and_nevis': 'KN', 'saint_lucia': 'LC', 'saint_martin_(french_part)': 'MF', 'saint_pierre_and_miquelon': 'PM', 'saint_vincent_and_the_grenadines': 'VC', 'samoa': 'WS', 'san_marino': 'SM', 'sao_tome_and_principe': 'ST', 'saudi_arabia': 'SA', 'senegal': 'SN', 'serbia': 'RS', 'seychelles': 'SC', 'sierra_leone': 'SL', 'singapore': 'SG', 'sint_maarten_(dutch_part)': 'SX', 'slovakia': 'SK', 'slovenia': 'SI', 'solomon_islands': 'SB', 'somalia': 'SO', 'south_africa': 'ZA', 'south_georgia_and_the_south_sandwich_islands': 'GS', 'south_sudan': 'SS', 'spain': 'ES', 'sri_lanka': 'LK', 'sudan': 'SD', 'suriname': 'SR', 'svalbard_and_jan_mayen': 'SJ', 'swaziland': 'SZ', 'sweden': 'SE', 'switzerland': 'CH', 'syrian_arab_republic': 'SY', 'taiwan,_province_of_china': 'TW', 'tajikistan': 'TJ', 'tanzania': 'TZ', 'thailand': 'TH', 'timor-leste': 'TL', 'togo': 'TG', 'tokelau': 'TK', 'tonga': 'TO', 'trinidad_and_tobago': 'TT', 'tunisia': 'TN', 'turkey': 'TR', 'turkmenistan': 'TM', 'turks_and_caicos_islands': 'TC', 'tuvalu': 'TV', 'uganda': 'UG', 'ukraine': 'UA', 'united_arab_emirates': 'AE', 'united_kingdom': 'GB', 'united_states': 'US', 'united_states_minor_outlying_islands': 'UM', 'uruguay': 'UY', 'uzbekistan': 'UZ', 'vanuatu': 'VU', 'venezuela,_bolivarian_republic_of': 'VE', 'viet_nam': 'VN', 'virgin_islands,_british': 'VG', 'virgin_islands,_u.s.': 'VI', 'wallis_and_futuna': 'WF', 'western_sahara': 'EH', 'yemen': 'YE', 'zambia': 'ZM', 'zimbabwe': 'ZW'}


# means and standard deviations are calculated over the entire dataset (train + val + test)

MEANS = {
    'BLUE':    0.06547681,
    'GREEN':   0.09543012,
    'RED':     0.10692262,
    'SWIR1':   0.22902039,
    'SWIR2':   0.15596166,
    'TEMP1': 298.51077,
    'NIR':     0.2542566,
    'DMSP':   41.69006032536221,
    'VIIRS':   3.443405293536357
    # 'NIGHTLIGHTS': 20.753946  # nightlights overall
}

STD_DEVS = {
    'BLUE':    0.031534348,
    'GREEN':   0.04290699,
    'RED':     0.06305032,
    'SWIR1':   0.09461683,
    'SWIR2':   0.08886013,
    'TEMP1':   8.608657,
    'NIR':     0.06909249,
    'DMSP':  109.7289863161683,
    'VIIRS':  10.97284745670896
    # 'NIGHTLIGHTS': 76.66724  # nightlights overall
}


def split_by_countries(idxs, ood_countries, metadata):
    countries = list(metadata['country'].iloc[idxs])
    countries = np.asarray([DHS_COUNTRY_MAP[country] for country in countries])
    is_ood = np.any([(countries == country) for country in ood_countries], axis=0)
    return idxs[~is_ood], idxs[is_ood]


class PovertyMapDataset(SustainBenchDataset):
    """The PovertyMap poverty measure prediction dataset.

    This is a processed version of LandSat 5/7/8 Surface Reflectance,
    DMSP-OLS, and VIIRS Nightlights satellite imagery originally
    from Google Earth Engine under the names
        Landsat 8: `LANDSAT/LC08/C01/T1_SR`
        Landsat 7: `LANDSAT/LE07/C01/T1_SR`
        Landsat 5: `LANDSAT/LT05/C01/T1_SR`
         DMSP-OLS: `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4`
            VIIRS: `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG`.
    The labels come from surveys conducted through the DHS Program:
        https://dhsprogram.com/data/available-datasets.cfm

    All of the images and surveys are processed in a manner similar to
        https://github.com/sustainlab-group/africa_poverty

    Supported `split_scheme`:
        'official' and `countries`, which are equivalent

    Input (x):
        224 x 224 x 8 satellite image, with 7 channels from Landsat and
        1 nighttime light channel from DMSP/VIIRS. These images have not been
        mean / std normalized.

    Output (y):
        y is a real-valued asset wealth index. Higher value corresponds to more
        asset wealth.

    Metadata:
        Each image is annotated with location coordinates (lat/lon, noised for
        anonymity), survey year, urban/rural classification, country.

    Website: https://github.com/sustainlab-group/africa_poverty

    Original publication:
    @article{yeh2020using,
        author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and
                  Azzari, George and Tang, Zhongyi and Lobell, David and
                  Ermon, Stefano and Burke, Marshall},
        day = {22},
        doi = {10.1038/s41467-020-16185-w},
        issn = {2041-1723},
        journal = {Nature Communications},
        month = {5},
        number = {1},
        title = {{Using publicly available satellite imagery and deep learning to
                  understand economic well-being in Africa}},
        url = {https://www.nature.com/articles/s41467-020-16185-w},
        volume = {11},
        year = {2020}
    }

    License:
        LandSat/DMSP/VIIRS data is U.S. Public Domain.
    """
    _dataset_name = 'poverty'
    _versions_dict = {
        '1.0': {
            'download_urls': [
                {'url': 'dhs_AL_DR.tar.gz',
                 'size': 16_472_693_417},
                {'url': 'dhs_EG_HT.tar.gz',
                 'size': 13_579_206_686},
                {'url': 'dhs_IA_IA.tar.gz',
                 'size': 24_046_259_399},
                {'url': 'dhs_ID_MZ.tar.gz',
                 'size': 18_386_761_224},
                {'url': 'dhs_NG_SZ.tar.gz',
                 'size': 18_911_963_362},
                {'url': 'dhs_TD_ZW.tar.gz',
                 'size':  9_024_655_370},
                {'url': 'dhs_final_labels.csv',
                 'size':     19_356_345}
            ]
        },
        '1.1': {'download_url': None}
    }


    def __init__(self, version=None, root_dir='data', download=False,
                 split_scheme='official',
                 no_nl=False, fold='A', oracle_training_set=False,
                 use_ood_val=True,
                 cache_size=100):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        # While the daytime image bands have a native 30m/pixel resolution, the nightlights images have a lower native resolution and are upsampled using the nearest-neighbors algorithm to match the daytime image resolution
        self.resolution = 30.
        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}

        if split_scheme=='official':
            split_scheme = 'countries'
        self._split_scheme = split_scheme
        if self._split_scheme != 'countries':
            raise ValueError("Split scheme not recognized")

        self.oracle_training_set = oracle_training_set

        self.no_nl = no_nl
        if fold not in {'A', 'B', 'C', 'D', 'E'}:
            raise ValueError("Fold must be A, B, C, D, or E")

        self.root = Path(self._data_dir)
        self.metadata = pd.read_csv(self.root / 'dhs_metadata.csv')
        self.metadata.rename(columns={"urban_rural": "urban"}, inplace=True)
        # country folds, split off OOD
        country_folds = SPLITS

        self._split_array = -1 * np.ones(len(self.metadata))
        incountry_folds_split = np.arange(len(self.metadata))
        # take the test countries to be ood
        idxs_id, idxs_ood_test = split_by_countries(incountry_folds_split, country_folds['test'], self.metadata)
        # also create a validation OOD set
        idxs_id, idxs_ood_val = split_by_countries(idxs_id, country_folds['val'], self.metadata)

        for split in ['test', 'val', 'id_test', 'id_val', 'train']:
            # keep ood for test, otherwise throw away ood data
            if split == 'test':
                idxs = idxs_ood_test
            elif split == 'val':
                idxs = idxs_ood_val
            else:
                idxs = idxs_id
                num_eval = 2000
                # if oracle, do 50-50 split between OOD and ID
                if split == 'train' and self.oracle_training_set:
                    idxs = subsample_idxs(incountry_folds_split, num=len(idxs_id), seed=ord(fold))[num_eval:]
                elif split != 'train' and self.oracle_training_set:
                    eval_idxs = subsample_idxs(incountry_folds_split, num=len(idxs_id), seed=ord(fold))[:num_eval]
                elif split == 'train':
                    idxs = subsample_idxs(idxs, take_rest=True, num=num_eval, seed=ord(fold))
                else:
                    eval_idxs  = subsample_idxs(idxs, take_rest=False, num=num_eval, seed=ord(fold))

                if split != 'train':
                    if split == 'id_val':
                        idxs = eval_idxs[:num_eval//2]
                    else:
                        idxs = eval_idxs[num_eval//2:]
            self._split_array[idxs] = self._split_dict[split]

        if not use_ood_val:
            self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
            self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

        self._y_array = torch.from_numpy(np.asarray(self.metadata['wealthpooled'])[:, np.newaxis]).float()
        self._y_size = 1

        # add country group field
        country_to_idx = {country: i for i, country in enumerate(DHS_COUNTRIES)}
        self.metadata['country'] = [country_to_idx[country] for country in self.metadata['country'].tolist()]
        self._metadata_map = {'country': DHS_COUNTRIES}
        self._metadata_array = torch.from_numpy(self.metadata[['urban', 'wealthpooled', 'country', 'lat', 'lon']].astype(float).to_numpy())
        # rename wealthpooled to y
        self._metadata_fields = ['urban', 'y', 'country', 'lat', 'lon']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['urban'])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{idx}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()

        return img

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model
            - y_true (LongTensor): Ground-truth values
            - metadata (Tensor): Metadata
            - prediction_fn (function): Only None supported
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        assert prediction_fn is None, "PovertyMapDataset.eval() does not support prediction_fn"

        metrics = [MSE(), PearsonCorrelation()]

        all_results = {}
        all_results_str = ''
        for metric in metrics:
            results, results_str = self.standard_group_eval(
                metric,
                self._eval_grouper,
                y_pred, y_true, metadata)
            all_results.update(results)
            all_results_str += results_str
        return all_results, all_results_str
