import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep
from itertools import combinations


class Neighborhood(MRJob):
    def steps(self):
        return [
                    MRStep(mapper=self.user_rating, reducer= self.user_count),
                    MRStep(mapper=self.remix, reducer = self.similarity),
                    MRStep(mapper=self.pseudo_matrix, reducer=self.save_matrix)
               ]


    def user_rating(self, key, line):
        user_id, movie_id, rate = line.split(',')
        yield user_id, (movie_id, rate)


    def user_count(self, user_id, values):
        user_rates = 0
        user_mean = 0
        final = []
        for movie_id, rate in values:
            r = float(rate) # / 5
            user_rates += 1
            user_mean += r
            final.append([int(movie_id), r])

        user_mean /= (user_rates + 1e-5)
        for i in range(len(final)):
            final[i][1] -= user_mean

        yield user_id, final


    def remix(self, user_id, values):
        for item1, item2 in combinations(values, 2):
            yield (item1[0], item2[0]), (item1[1], item2[1])


    def similarity(self, pair_id, rates):
        sum_x = 0
        sum_y = 0
        sum_xy = 0
        for rate_x, rate_y in rates:
            sum_x += rate_x ** 2
            sum_y += rate_y ** 2
            sum_xy += rate_x * rate_y
        simila = sum_xy / (np.sqrt(sum_x) * np.sqrt(sum_y))
        if simila < 0:
            simila = 0
        yield pair_id, simila


    def pseudo_matrix(self, pair_id, sim):
        yield pair_id[0], (pair_id[1], sim)
        yield pair_id[1], (pair_id[0], sim)


    def save_matrix(self, movie_id, relate_movie):
        related = []
        for item in relate_movie:
            related.append(item)
        related.sort(key= lambda x: x[1], reverse=True)
        yield movie_id, related


if __name__ == '__main__':
    Neighborhood.run()
