class DDStreamModel:
    def __init__(self, broadcasted_var, ssc):
        self.eps = 0.01
        self.broadcasted_var = broadcasted_var
        self.scc = ssc

    def assign_to_micro_cluster(self, row):
        if len(self.broadcasted_var.value) > 5:
            print("\n\nHERE\n\n")
        print(f"row assigned to micro cluster {row}")
        return (0, row)

    def update_micro_clusters(self, assignations):
        print(f"update_mc {assignations}")
        pass

    def run_batch(self, df, batch_id):
        print(batch_id, df, end="\n")
        self.broadcasted_var 

    def run(self, row):
        initialized = True
        if row and initialized:
            print(row)
            assignations = self.assign_to_micro_cluster(row)
            self.update_micro_clusters(assignations)
