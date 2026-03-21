from descriptors.soap_generator import SOAPFingerPrintGenerator
from descriptors.acsf_generator import ACSFGenerator


class DescriptorFactory:
    @staticmethod
    def create(config):
        descriptor_type = config.descriptor.get('type', 'soap').lower()
        
        if descriptor_type == 'soap':
            return SOAPFingerPrintGenerator(
                r_cut=config.soap['r_cut'],
                n_max=config.soap['n_max'],
                l_max=config.soap['l_max'],
                allowed_species=config.soap['allowed_species']
            )
        elif descriptor_type == 'acsf':
            return ACSFGenerator(
                r_cut=config.acsf['r_cut'],
                g2_params=config.acsf.get('g2_params'),
                g4_params=config.acsf.get('g4_params'),
                allowed_species=config.acsf['allowed_species']
            )
        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")