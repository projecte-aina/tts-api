import argparse
import multiprocessing as mp
class MpWorkersAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if int(values) > mp.cpu_count():
            parser.error("Maximum value for {0} is {1}".format(option_string, mp.cpu_count()))
            #raise argparse.ArgumentError("Minimum bandwidth is 12")
        if int(values) <= 1:
            print(values)
            parser.error("Minimum value for {0} is 2".format(option_string))

        setattr(namespace, self.dest, values)
