from pyspark.sql.streaming import StreamingQueryListener


class MyListener(StreamingQueryListener):
    def onQueryStarted(self, event):
        pass

    def onQueryProgress(self, event):
        """
        Called when there is some status update (ingestion rate updated, etc.)

        Parameters
        ----------
        event: :class:`pyspark.sql.streaming.listener.QueryProgressEvent`
            The properties are available as the same as Scala API.

        Notes
        -----
        This method is asynchronous. The status in
        :class:`pyspark.sql.streaming.StreamingQuery` will always be
        latest no matter when this method is called. Therefore, the status
        of :class:`pyspark.sql.streaming.StreamingQuery`.
        may be changed before/when you process the event.
        For example, you may find :class:`StreamingQuery`
        is terminated when you are processing `QueryProgressEvent`.
        """
        print(f"HERE: Query progress\n {event.progress.json}\n")

    def onQueryTerminated(self, event):
        pass


my_listener = MyListener()
