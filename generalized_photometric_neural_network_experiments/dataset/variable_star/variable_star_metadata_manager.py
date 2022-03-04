from pathlib import Path

import pandas as pd
from astroquery.vizier import Vizier

from generalized_photometric_neural_network_experiments.dataset.variable_star.download_metadata import has_gcvs_type, \
    GcvsColumnName, get_tic_id_for_gcvs_row, download_gaia_metadata_csv, get_tic_id_for_gaia_row
from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from peewee import IntegerField, SchemaManager, CharField, IntegrityError

from ramjet.data_interface.metadatabase import MetadatabaseModel, metadatabase


class VariableTypeName(Enum):
    RR_LYRAE = 'rr_lyrae'
    OTHER = 'other'


class VariableStarMetadata(MetadatabaseModel):
    tic_id = IntegerField(index=True, unique=True)
    variable_type_name = CharField(index=True, choices=VariableTypeName)

    class Meta:
        """Schema meta data for the model."""
        indexes = (
            (('variable_type_name', 'tic_id'), False),
        )


class VariableStarMetadataManager:
    @staticmethod
    def build_table_with_gcvs_data():
        print('Building variable star metadata table...')
        row_count = 0
        duplicate_count = 0
        metadatabase.drop_tables([VariableStarMetadata])
        metadatabase.create_tables([VariableStarMetadata])
        gcvs_catalog_astropy_table = Vizier(columns=['**'], catalog='B/gcvs/gcvs_cat', row_limit=-1
                                            ).query_constraints()[0]
        gcvs_catalog_data_frame = gcvs_catalog_astropy_table.to_pandas()
        with metadatabase.atomic():
            for row_index, row in gcvs_catalog_data_frame.iterrows():
                rr_lyrae_labels = ['RR', 'RR(B)', 'RRAB', 'RRC']
                is_lyrae = has_gcvs_type(row[GcvsColumnName.VARIABLE_TYPE_STRING], rr_lyrae_labels)
                if is_lyrae:
                    variable_type_name = VariableTypeName.RR_LYRAE.value
                else:
                    variable_type_name = VariableTypeName.OTHER.value
                tic_id = get_tic_id_for_gcvs_row(row)
                if tic_id is None:
                    continue
                row = VariableStarMetadata(tic_id=tic_id, variable_type_name=variable_type_name)
                try:
                    row.save()
                except IntegrityError:
                    duplicate_count += 1
                row_count += 1
                print(row_count)
        print(f'Table built. {row_count} rows added. {duplicate_count} unadded duplicates.')

    @staticmethod
    def build_table_with_gaia_data():
        print('Building variable star metadata table...')
        row_count = 0
        duplicate_count = 0
        metadatabase.drop_tables([VariableStarMetadata])
        metadatabase.create_tables([VariableStarMetadata])
        gaia_variable_targets_csv_path = Path('data/variables/gaia_variable_targets.csv')
        gaia_variable_targets_csv_path.parent.mkdir(exist_ok=True, parents=True)
        if not gaia_variable_targets_csv_path.exists():
            download_gaia_metadata_csv()
        gaia_variable_targets_data_frame = pd.read_csv(gaia_variable_targets_csv_path, index_col=False)
        with metadatabase.atomic():
            for row_index, row in gaia_variable_targets_data_frame.iterrows():
                rr_lyrae_labels = ['ARRD', 'RRC', 'RRAB', 'RRD']
                is_lyrae = row['best_class_name'] in rr_lyrae_labels
                if is_lyrae:
                    variable_type_name = VariableTypeName.RR_LYRAE.value
                else:
                    variable_type_name = VariableTypeName.OTHER.value
                tic_id = get_tic_id_for_gaia_row(row)
                if tic_id is None:
                    continue
                row = VariableStarMetadata(tic_id=tic_id, variable_type_name=variable_type_name)
                try:
                    row.save()
                except IntegrityError:
                    duplicate_count += 1
                row_count += 1
                print(row_count)
        print(f'Table built. {row_count} rows added. {duplicate_count} unadded duplicates.')


if __name__ == '__main__':
    metadata_manager = VariableStarMetadataManager()
    metadata_manager.build_table_with_gaia_data()
