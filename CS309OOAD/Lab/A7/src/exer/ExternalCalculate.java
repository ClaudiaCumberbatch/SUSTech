package exer;

public class ExternalCalculate extends Calculate {
    public int calculateParkingPrice(double hours) {
        int price = 0;
        price += (int) (hours / 24) * 60;
        hours = hours % 24;
        if (hours >= 0.5 && hours < 1) {
            price += 15;
        } else if (hours >= 1) {
            price = Math.min(15 + 5 * ((int) hours - 1), 60);
        }
        return price;
    }

    @Override
    public int calculatePoints(double hours) {
        return (int) hours / 24 * 2 + 1;
    }
}
