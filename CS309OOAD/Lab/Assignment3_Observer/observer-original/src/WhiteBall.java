import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class WhiteBall extends Ball implements Subject<Ball>{
    public WhiteBall(int xSpeed, int ySpeed, int ballSize) {
        super(Color.WHITE, xSpeed, ySpeed, ballSize);
    }

    @Override
    public void update(char keyChar) {
//        System.out.println("WhiteBall update");
        switch (keyChar) {
            case 'a':
                this.setXSpeed(-8);
                break;
            case 'd':
                this.setXSpeed(8);
                break;
            case 'w':
                this.setYSpeed(-8);
                break;
            case 's':
                this.setYSpeed(8);
                break;
        }
    }

    @Override
    public void update(Ball ball) {

    }

    public void move() {
        int newX = this.getX() + this.getXSpeed();
        int newY = this.getY() + this.getYSpeed();

        this.setX(newX);
        this.setY(newY);

        if (newX <= 0) {
            this.setXSpeed(Math.abs(getXSpeed()));
        } else if (newX >= 600 - this.getBallSize()) {
            this.setXSpeed(-1 * Math.abs(getXSpeed()));
        }

        if (newY <= 0) {
            this.setYSpeed(Math.abs(getYSpeed()));
        } else if (newY > 600 - this.getBallSize()) {
            this.setYSpeed(-1 * Math.abs(getYSpeed()));
        }

        this.notifyObservers();
    }

    private List<Ball> observerList = new ArrayList<>();

    @Override
    public void registerObserver(Ball ball) {
        observerList.add(ball);
    }

    @Override
    public void removeObserver(Ball ball) {
        observerList.remove(ball);
    }

    @Override
    public void notifyObservers(char keyChar) {

    }

    @Override
    public void notifyObservers() {
        observerList.forEach(ball -> ball.update(this));
    }
}
