import pandas_profiling

def profiling_reports(data, title):
    """Create pandas profiling report

    :param data: Data to create report for
    :type data: DataFrame
    :param title: Title of Report
    :type title: str
    """
    profile = pandas_profiling.ProfileReport(data, title=title,  explorative=True)
    profile.to_file("report_" + title + ".html")



