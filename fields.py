import logging

import torchtext

class AttentionField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True, use_vocab to be false, and applies postprocessing to integers
    Since we already define the attention vectors with integers in the data set, we don't need a vocabulary. Instead, we directly use the provided integers
    """

    def __init__(self, ignore_index, **kwargs):
        """
        Initialize the AttentionField. As pre-processing it prepends the ignore value, to account for the SOS step
        
        Args:
            ignore_index (int): The value that will be ignored for metric and loss calculation, when using attention loss
            **kwargs: The extra arguments for the parent class 
        """
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use machine. Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('use_vocab') == True:
            logger.warning("Option use_vocab has to be set to False for the attention field. Changed to False")
        kwargs['use_vocab'] = False

        if kwargs.get('preprocessing') is not None:
            logger.error("No pre-processing allowed for the attention field")

        def preprocess(seq):
            return [self.ignore_index] + seq

        if kwargs.get('postprocessing') is not None:
            logger.error("No post-processing allowed for the attention field")

        # Post-processing function receives batch and positional arguments(?).
        # Batch is a 2D list with batch examples in dim-0 and sequences in dim-1
        # For each element in each example we convert from unicode string to integer.
        # PAD is converted to -1
        def post_process_function(example, __):
            def safe_cast(cast_func, x, default):
                try:
                    return cast_func(x)
                except (ValueError, TypeError):
                    return default

            return [safe_cast(int, item, self.ignore_index) for item in example]

        post_process_pipeline = torchtext.data.Pipeline(convert_token=post_process_function)
        
        kwargs['preprocessing'] = preprocess
        kwargs['postprocessing'] = post_process_pipeline


        super(AttentionField, self).__init__(**kwargs)

        self.ignore_index = ignore_index
